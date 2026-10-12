package native

import (
	"context"
	"fmt"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
)

// Pack IDs of the known configurations. A pack directory is expected at
// <models-dir>/<pack-id>/.
const (
	PackNomicEmbedTextV15 = "nomic-embed-text-v1.5"
	PackBGESmallENV15     = "bge-small-en-v1.5"
	// PackArcticEmbedMLong is nomic-bert architecture (RoPE θ=1000 + SwiGLU,
	// no attention/FFN biases) with CLS pooling — config-only on the existing
	// encoder. Upstream declares 8192 positions via dynamic-NTK RoPE scaling
	// (rotary_scaling_factor 2), which only engages PAST the trained 2048;
	// the pack pins max_trained_positions 2048, so the standard trained-
	// context clamp keeps the engine exactly in upstream's unscaled regime.
	PackArcticEmbedMLong = "snowflake-arctic-embed-m-long"
)

// Provider runs a native encoder behind futon's Provider
// interface (implemented structurally — this package does not import
// internal/embed).
//
// Prefix policy: NONE. futon embeds raw text
// (no search_query:/search_document: prefixes anywhere in the tree), so the
// native provider embeds raw text too — continuity with every prefix-less vector
// a caller has already stored. Pack manifests may carry the upstream prefix
// templates for provenance, but they are never applied.
type Provider struct {
	model *Model
	tok   *Tokenizer
	dims  int

	// threads is the TOTAL kernel thread budget (Options.Threads resolved,
	// ≥ 1). Single embeds use all of it; EmbedBatch divides it across its
	// worker pool so batchWorkers concurrent forwards never demand
	// batchWorkers×threads goroutines (MED-1).
	threads int

	// batchWorkers caps concurrent per-text forwards inside EmbedBatch.
	batchWorkers int

	mu     sync.Mutex
	closed bool
}

// Options configures a native Provider.
type Options struct {
	// PackDir is the weight-pack directory (config.json, model.safetensors,
	// vocab.txt). Required — Phase 1 never downloads anything.
	PackDir string
	// Dimensions requests matryoshka truncation to this output width.
	// 0 = the model's full width. Non-zero values require pack support.
	Dimensions int
	// Threads is the kernel thread budget for a single forward pass.
	// 0 = all CPUs.
	Threads int
}

// New loads the pack at opts.PackDir and returns a ready Provider.
// It validates that a requested Dimensions is achievable so misconfiguration
// fails at startup, not at first embed.
func New(opts Options) (*Provider, error) {
	if opts.PackDir == "" {
		return nil, fmt.Errorf("native: PackDir is required (no downloading in this phase)")
	}
	model, err := LoadModel(opts.PackDir, opts.Threads)
	if err != nil {
		return nil, err
	}
	cfg := model.Config()
	tok, err := NewTokenizer(filepath.Join(opts.PackDir, "vocab.txt"), cfg.Tokenizer)
	if err != nil {
		return nil, err
	}
	if tok.VocabSize() > cfg.VocabSize {
		return nil, fmt.Errorf("native: vocab.txt has %d entries but pack declares vocab_size %d", tok.VocabSize(), cfg.VocabSize)
	}
	dims := opts.Dimensions
	if dims == 0 {
		dims = cfg.HiddenSize
	}
	if dims != cfg.HiddenSize {
		if cfg.Matryoshka == nil {
			return nil, fmt.Errorf("native: pack %s does not support truncation to %d dimensions", cfg.ModelID, dims)
		}
		if dims < 1 || dims > cfg.HiddenSize {
			return nil, fmt.Errorf("native: dimensions %d outside [1,%d]", dims, cfg.HiddenSize)
		}
	}
	threads := opts.Threads
	if threads < 1 {
		threads = runtime.NumCPU()
	}
	return &Provider{
		model:   model,
		tok:     tok,
		dims:    dims,
		threads: threads,
		// One worker per budgeted thread: with the per-forward split below,
		// total kernel parallelism stays ≈ threads regardless of batch size.
		batchWorkers: threads,
	}, nil
}

// Embed generates one embedding. Uses the full kernel thread budget for the
// single forward pass.
func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	ids := p.tok.Encode(text, p.model.Config().EffectiveMaxSeqLen())
	return p.model.EncodeTokens(ctx, ids, p.dims)
}

// batchPlan sizes EmbedBatch's resources: a worker pool of exactly
// min(batchWorkers, nTexts) goroutines (never goroutine-per-text), each
// running forwards on a kernel using ~threads/workers goroutines so total
// kernel parallelism stays ≈ threads instead of workers×threads (MED-1).
func batchPlan(nTexts, batchWorkers, threads int) (workers, perForward int) {
	workers = batchWorkers
	if workers > nTexts {
		workers = nTexts
	}
	if workers < 1 {
		workers = 1
	}
	perForward = threads / workers
	if perForward < 1 {
		perForward = 1
	}
	return workers, perForward
}

// EmbedBatch embeds texts through a bounded worker pool (≤ the provider's
// thread budget, never one goroutine per text). Results are index-aligned;
// the whole batch fails on the first error (native inference has no
// per-item transient failures worth tolerating).
//
// Batch results are bit-identical to calling Embed per text: each text runs
// an independent unpadded forward, so batching cannot perturb pooling or
// attention (no padding, no masks — see Model.EncodeTokens), and the
// per-forward thread split cannot change output bits (the kernel
// determinism contract; pinned by TestBatchMatchesSingleAcrossWorkerCounts).
func (p *Provider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if err := p.checkOpen(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return nil, nil
	}
	maxSeq := p.model.Config().EffectiveMaxSeqLen()

	workers, _ := batchPlan(len(texts), p.batchWorkers, p.threads)
	kern := newElasticKernel(p.threads, workers, len(texts))

	result := make([][]float32, len(texts))
	errs := make([]error, len(texts))
	next := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				if err := ctx.Err(); err != nil {
					errs[i] = err
					continue
				}
				ids := p.tok.Encode(texts[i], maxSeq)
				result[i], errs[i] = p.model.EncodeTokensKernel(ctx, ids, p.dims, kern)
				kern.remaining.Add(-1)
			}
		}()
	}
	// Longest first: a long text dispatched last would run alone while the
	// other workers idle. Byte length is a cheap proxy for token count.
	// Order affects only scheduling — every result is computed independently.
	order := make([]int, len(texts))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return len(texts[order[a]]) > len(texts[order[b]]) })
	for _, i := range order {
		next <- i
	}
	close(next)
	wg.Wait()
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("native: batch item %d: %w", i, err)
		}
	}
	return result, nil
}

// elasticKernel shares a batch's thread budget among its in-flight
// forwards. Each call runs on total/min(workers, remaining) threads, read at
// call time, so as the queue drains the last forwards (the longest, given
// longest-first dispatch) widen onto the idle cores instead of finishing
// single-threaded. Thread count never changes an output bit (the kernel
// determinism contract), so this is invisible in results.
type elasticKernel struct {
	byThreads []Kernel // byThreads[t] uses t threads, t in [1, total]
	workers   int64
	remaining atomic.Int64 // forwards not yet finished
}

func newElasticKernel(total, workers, n int) *elasticKernel {
	e := &elasticKernel{byThreads: make([]Kernel, total+1), workers: int64(workers)}
	for t := 1; t <= total; t++ {
		e.byThreads[t] = newKernel(t)
	}
	e.remaining.Store(int64(n))
	return e
}

func (e *elasticKernel) current() Kernel {
	inFlight := max(1, min(e.workers, e.remaining.Load()))
	return e.byThreads[max(1, int64(len(e.byThreads)-1)/inFlight)]
}

func (e *elasticKernel) MatMulT(dst, a, b []float32, m, n, k int, bias []float32) {
	e.current().MatMulT(dst, a, b, m, n, k, bias)
}

func (e *elasticKernel) MatMulTF16(dst, a []float32, b []uint16, m, n, k int, bias []float32) {
	e.current().MatMulTF16(dst, a, b, m, n, k, bias)
}

func (e *elasticKernel) MatMulTQ8(dst, a []float32, qb []int8, scales []float32, m, n, k int, bias []float32) {
	e.current().(q8Kernel).MatMulTQ8(dst, a, qb, scales, m, n, k, bias)
}

func (e *elasticKernel) threadCount() int { return kernelThreads(e.current()) }

func (e *elasticKernel) withThreads(n int) Kernel {
	return e.byThreads[max(1, min(n, len(e.byThreads)-1))]
}

// Dimensions returns the configured output width.
func (p *Provider) Dimensions() int { return p.dims }

// ModelName returns the ENGINE-QUALIFIED pack id, e.g.
// "native/nomic-embed-text-v1.5". Stored alongside embeddings for
// provenance. The "native/" prefix exists because the daemon does NOT
// faithfully implement nomic (llama.cpp tokenizer non-conformance — 15/26
// corpus texts below the 0.999 parity gate), so daemon vectors and native
// vectors are DIFFERENT spaces and must never share a name: a user can
// configure a server model literally called "nomic-embed-text-v1.5", but
// never one this string can collide with (an HTTP provider could refuse
// "native/"-prefixed model names, making the two directions provably
// disjoint).
func (p *Provider) ModelName() string { return NativeModelNamePrefix + p.model.Config().ModelID }

// NativeModelNamePrefix qualifies every native ModelName. See ModelName.
const NativeModelNamePrefix = "native/"

// IsAvailable reports whether the provider is usable (weights loaded and not
// closed). There is no daemon to probe.
func (p *Provider) IsAvailable(_ context.Context) bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return !p.closed
}

// Close marks the provider closed. Weight memory is released when the last
// in-flight forward finishes and the Provider is garbage collected.
func (p *Provider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	return nil
}

func (p *Provider) checkOpen() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return fmt.Errorf("native: provider is closed")
	}
	return nil
}
