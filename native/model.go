package native

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
)

// Model is a loaded BERT-family encoder: config + float32 weights + kernel.
//
// Weights are read-only after Load; every EncodeTokens call takes its own
// scratch buffers from a pool, so a Model is safe for concurrent use.
//
// Determinism: for a given pack and host, EncodeTokens is bit-deterministic
// regardless of kernel thread count — every output element of every GEMM is
// one serial dot product with a fixed reduction order; goroutines only decide
// WHICH elements a worker computes, never the order WITHIN an element, and
// all other reductions (softmax, layernorm, pooling) are serial loops.
// Bit-equality across DIFFERENT hosts/architectures is not promised (libm
// and FMA differences); parity there is guarded by the cosine gates instead.
type Model struct {
	cfg  *PackConfig
	kern Kernel

	headDim int

	// Embedding tables. The word table is the largest tensor in a pack, so
	// it stays in the pack's storage dtype (fp16 for real packs) and rows
	// are converted during the gather; the small tables are float32.
	wordEmb  weightMat // [vocab, hidden]
	posEmb   []float32 // [maxpos, hidden] — absolute only
	typeEmb  []float32 // [types, hidden] — optional
	embNormW []float32
	embNormB []float32

	layers []layerWeights

	// Rotary tables, precomputed eagerly at load for capacity
	// EffectiveMaxSeqLen (the trained context — positions past it are
	// refused, never extrapolated), keeping concurrent use lock-free.
	ropeCos []float32 // [effmaxseq, headDim/2]
	ropeSin []float32

	scratch sync.Pool // *forwardScratch, reused across forwards
}

// weightMat is a matrix in whichever dtype the pack stores: raw fp16 bits
// (real packs — half the RAM, and the SIMD kernel converts in-register) or
// float32 (toy packs). Exactly one of f16/f32 is non-nil. q8 is only set on
// the experimental FUTON_NATIVE_Q8=1 path (see quant.go), in which case it
// REPLACES the float storage for GEMM weights.
type weightMat struct {
	f16 []uint16
	f32 []float32
	q8  *q8Mat
}

// rowInto copies one row of cols elements into dst, converting fp16 exactly.
func (w weightMat) rowInto(dst []float32, row, cols int) {
	if w.f16 != nil {
		src := w.f16[row*cols : row*cols+cols]
		for i, h := range src {
			dst[i] = f16ToF32(h)
		}
		return
	}
	copy(dst, w.f32[row*cols:row*cols+cols])
}

// mmT dispatches dst = a·wᵀ + bias to the dtype-matching kernel entry point.
func mmT(kern Kernel, dst, a []float32, w weightMat, rows, n, k int, bias []float32) {
	if w.q8 != nil {
		// Experimental Q8 path (quant.go). Every shipped kernel implements
		// q8Kernel; the assertion is a guard against a future kernel that
		// forgets — quantized weights without a Q8 kernel must be loud,
		// never a silent fallthrough into reading nil float storage.
		kern.(q8Kernel).MatMulTQ8(dst, a, w.q8.data, w.q8.scale, rows, n, k, bias)
		return
	}
	if w.f16 != nil {
		kern.MatMulTF16(dst, a, w.f16, rows, n, k, bias)
		return
	}
	kern.MatMulT(dst, a, w.f32, rows, n, k, bias)
}

type layerWeights struct {
	qkvW           weightMat // [3*hidden, hidden] — q rows, then k, then v (heads-major)
	qkvB           []float32 // optional
	outW           weightMat // [hidden, hidden]
	outB           []float32 // optional
	norm1W, norm1B []float32
	fc1W           weightMat // [inter, hidden] (swiglu: the value/up projection, nomic fc11)
	fc1B           []float32 // optional
	gateW          weightMat // [inter, hidden] — swiglu only (nomic fc12, receives silu)
	gateB          []float32 // optional
	fc2W           weightMat // [hidden, inter]
	fc2B           []float32 // optional
	norm2W, norm2B []float32
}

// LoadModel loads a weight pack from dir with the given kernel thread count
// (threads < 1 = all CPUs).
//
// Every pack member is verified against manifest.json before use (LOW-2):
// per-member sha256 + size, plus the exact tensor-name set when the
// manifest records one. Hashing the weights costs ~1s for the 274MB nomic
// pack, once per load — see loadAndVerifyManifest.
func LoadModel(dir string, threads int) (*Model, error) {
	man, err := loadAndVerifyManifest(dir)
	if err != nil {
		return nil, err
	}
	cfg, err := LoadPackConfig(dir)
	if err != nil {
		return nil, err
	}
	if man.ModelID != cfg.ModelID {
		return nil, fmt.Errorf("pack %s: manifest is for %q — mixed pack contents", cfg.ModelID, man.ModelID)
	}
	tensors, err := LoadSafetensorsRaw(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, fmt.Errorf("pack %s: %w", cfg.ModelID, err)
	}
	if err := man.verifyTensorNames(tensors); err != nil {
		return nil, err
	}
	return newModel(cfg, tensors, threads)
}

func newModel(cfg *PackConfig, tensors map[string]*Tensor, threads int) (*Model, error) {
	m := &Model{
		cfg:     cfg,
		kern:    newKernel(threads),
		headDim: cfg.HiddenSize / cfg.NumHeads,
	}
	h, inter := cfg.HiddenSize, cfg.IntermediateSize

	used := make(map[string]bool, len(tensors))
	needT := func(name string, shape ...int) (*Tensor, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("pack %s: missing tensor %q", cfg.ModelID, name)
		}
		used[name] = true
		if len(t.Shape) != len(shape) {
			return nil, fmt.Errorf("pack %s: tensor %q rank %d, want %d", cfg.ModelID, name, len(t.Shape), len(shape))
		}
		for i, d := range shape {
			if t.Shape[i] != d {
				return nil, fmt.Errorf("pack %s: tensor %q shape %v, want %v", cfg.ModelID, name, t.Shape, shape)
			}
		}
		return t, nil
	}
	// needMat keeps the pack's dtype (fp16 stays fp16 — the kernels consume
	// it directly); need/opt dequantize to float32 for the small tensors
	// (norms, biases, compact embedding tables).
	needMat := func(name string, shape ...int) (weightMat, error) {
		t, err := needT(name, shape...)
		if err != nil {
			return weightMat{}, err
		}
		if t.F16 != nil {
			return weightMat{f16: t.F16}, nil
		}
		return weightMat{f32: t.Data}, nil
	}
	need := func(name string, shape ...int) ([]float32, error) {
		t, err := needT(name, shape...)
		if err != nil {
			return nil, err
		}
		return tensorF32(t), nil
	}
	opt := func(name string, shape ...int) ([]float32, error) {
		if _, ok := tensors[name]; !ok {
			return nil, nil
		}
		return need(name, shape...)
	}

	var err error
	fail := func(e error) (*Model, error) { return nil, e }

	if m.wordEmb, err = needMat("embeddings.word", cfg.VocabSize, h); err != nil {
		return fail(err)
	}
	if cfg.PositionEncoding == PosAbsolute {
		if m.posEmb, err = need("embeddings.position", cfg.MaxSeqLen, h); err != nil {
			return fail(err)
		}
	}
	if cfg.TypeVocabSize > 0 {
		if m.typeEmb, err = need("embeddings.token_type", cfg.TypeVocabSize, h); err != nil {
			return fail(err)
		}
	}
	if m.embNormW, err = need("embeddings.norm.weight", h); err != nil {
		return fail(err)
	}
	if m.embNormB, err = need("embeddings.norm.bias", h); err != nil {
		return fail(err)
	}

	m.layers = make([]layerWeights, cfg.NumLayers)
	for i := range m.layers {
		lw := &m.layers[i]
		p := func(s string) string { return fmt.Sprintf("layers.%d.%s", i, s) }
		if lw.qkvW, err = needMat(p("attn.qkv.weight"), 3*h, h); err != nil {
			return fail(err)
		}
		if lw.qkvB, err = opt(p("attn.qkv.bias"), 3*h); err != nil {
			return fail(err)
		}
		if lw.outW, err = needMat(p("attn.out.weight"), h, h); err != nil {
			return fail(err)
		}
		if lw.outB, err = opt(p("attn.out.bias"), h); err != nil {
			return fail(err)
		}
		if lw.norm1W, err = need(p("norm1.weight"), h); err != nil {
			return fail(err)
		}
		if lw.norm1B, err = need(p("norm1.bias"), h); err != nil {
			return fail(err)
		}
		if lw.fc1W, err = needMat(p("mlp.fc1.weight"), inter, h); err != nil {
			return fail(err)
		}
		if lw.fc1B, err = opt(p("mlp.fc1.bias"), inter); err != nil {
			return fail(err)
		}
		if cfg.Activation == ActSwiGLU {
			if lw.gateW, err = needMat(p("mlp.gate.weight"), inter, h); err != nil {
				return fail(err)
			}
			if lw.gateB, err = opt(p("mlp.gate.bias"), inter); err != nil {
				return fail(err)
			}
		}
		if lw.fc2W, err = needMat(p("mlp.fc2.weight"), h, inter); err != nil {
			return fail(err)
		}
		if lw.fc2B, err = opt(p("mlp.fc2.bias"), h); err != nil {
			return fail(err)
		}
		if lw.norm2W, err = need(p("norm2.weight"), h); err != nil {
			return fail(err)
		}
		if lw.norm2B, err = need(p("norm2.bias"), h); err != nil {
			return fail(err)
		}
	}

	for name := range tensors {
		if !used[name] {
			return fail(fmt.Errorf("pack %s: unexpected tensor %q (pack/config mismatch)", cfg.ModelID, name))
		}
	}

	// Experimental Q8 (quant.go, OFF by default): when FUTON_NATIVE_Q8=1,
	// re-quantize the layer GEMM weights to per-row int8 and drop their
	// float storage. The embedding tables stay float — the word table is
	// gathered row-wise, never multiplied — and norms/biases are tiny.
	if os.Getenv(envNativeQ8) == "1" {
		for i := range m.layers {
			lw := &m.layers[i]
			for _, spec := range []struct {
				w    *weightMat
				n, k int
			}{
				{&lw.qkvW, 3 * h, h},
				{&lw.outW, h, h},
				{&lw.fc1W, inter, h},
				{&lw.gateW, inter, h},
				{&lw.fc2W, h, inter},
			} {
				if spec.w.f16 == nil && spec.w.f32 == nil {
					continue // absent optional (gateW on non-swiglu packs)
				}
				spec.w.q8 = quantizeQ8(*spec.w, spec.n, spec.k)
				spec.w.f16, spec.w.f32 = nil, nil
			}
		}
	}

	if cfg.PositionEncoding == PosRoPE {
		m.precomputeRoPE(cfg.EffectiveMaxSeqLen())
	}
	return m, nil
}

// Config returns the pack's configuration (read-only).
func (m *Model) Config() *PackConfig { return m.cfg }

// SetKernelThreads replaces the kernel with one using the given thread count.
// Not safe to call concurrently with EncodeTokens.
func (m *Model) SetKernelThreads(threads int) { m.kern = newKernel(threads) }

// precomputeRoPE fills cos/sin tables for positions [0, maxSeq) using
// float64 positions (matching HF's pos_idx_in_fp32) and the pack's theta.
func (m *Model) precomputeRoPE(maxSeq int) {
	half := m.headDim / 2
	m.ropeCos = make([]float32, maxSeq*half)
	m.ropeSin = make([]float32, maxSeq*half)
	for t := 0; t < maxSeq; t++ {
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(m.cfg.RoPETheta, float64(2*i)/float64(m.headDim))
			angle := float64(t) * freq
			m.ropeCos[t*half+i] = float32(math.Cos(angle))
			m.ropeSin[t*half+i] = float32(math.Sin(angle))
		}
	}
}

// EncodeTokens runs the forward pass over one token-id sequence (already
// wrapped in [CLS]/[SEP] and truncated to MaxSeqLen) and returns the pooled,
// optionally truncated (matryoshka), optionally L2-normalized embedding.
//
// outDim 0 means the model's full hidden size. Any other value requires the
// pack to declare matryoshka support and applies the pack's truncation
// recipe (parameter-free layernorm → slice → renormalize for nomic).
//
// Every token in ids is treated as real content: the provider never pads
// (each sequence runs unpadded, batch-of-one), so no attention mask exists
// to get wrong — padding cannot leak into attention or mean pooling by
// construction. Callers must not pass [PAD] ids.
//
// ctx is checked between layers and at intra-layer checkpoints;
// cancellation aborts with ctx.Err().
func (m *Model) EncodeTokens(ctx context.Context, ids []int, outDim int) ([]float32, error) {
	return m.EncodeTokensKernel(ctx, ids, outDim, m.kern)
}

// EncodeTokensKernel is EncodeTokens with an explicit kernel — the
// per-forward parallelism hint. EmbedBatch uses it to divide the total
// thread budget across concurrent forwards (each forward's GEMMs run on
// ~threads/batchWorkers goroutines) instead of oversubscribing
// batchWorkers×threads. The determinism contract makes the knob invisible
// in results: kernel thread count never changes any output bit (pinned by
// TestKernelDeterministicAcrossThreads and TestBatchMatchesSingle).
func (m *Model) EncodeTokensKernel(ctx context.Context, ids []int, outDim int, kern Kernel) ([]float32, error) {
	cfg := m.cfg
	if len(ids) == 0 {
		return nil, fmt.Errorf("native: empty token sequence")
	}
	if len(ids) > cfg.EffectiveMaxSeqLen() {
		return nil, fmt.Errorf("native: sequence length %d exceeds effective max_seq_len %d (declared %d, trained %d)",
			len(ids), cfg.EffectiveMaxSeqLen(), cfg.MaxSeqLen, cfg.MaxTrainedPositions)
	}
	if outDim < 0 || outDim > cfg.HiddenSize {
		return nil, fmt.Errorf("native: requested dim %d outside [1,%d]", outDim, cfg.HiddenSize)
	}
	if outDim != 0 && outDim != cfg.HiddenSize && cfg.Matryoshka == nil {
		return nil, fmt.Errorf("native: pack %s does not support matryoshka truncation to %d", cfg.ModelID, outDim)
	}
	for _, id := range ids {
		if id < 0 || id >= cfg.VocabSize {
			return nil, fmt.Errorf("native: token id %d outside vocab of %d", id, cfg.VocabSize)
		}
	}

	seq, h := len(ids), cfg.HiddenSize

	// Attention runs head-parallel when the kernel has threads to spare and
	// each head's GEMMs are too small to parallelize on their own (typical
	// embedding lengths); every head then runs its GEMMs on a serial copy of
	// the kernel. Longer sequences keep heads serial with parallel GEMMs,
	// which also bounds scratch to one head's seq² buffer.
	threads := kernelThreads(kern)
	attn := attnPlan{kern: kern, workers: 1}
	if t, ok := kern.(threaded); ok && threads > 1 && seq*padSeq(seq)*m.headDim < minParallelWork {
		attn.workers = min(threads, cfg.NumHeads)
		attn.kern = t.withThreads(1)
	}
	scratch := m.getScratch(seq, attn.workers)
	defer m.scratch.Put(scratch)

	// ── Embeddings ──
	x := scratch.x
	for t, id := range ids {
		row := x[t*h : t*h+h]
		m.wordEmb.rowInto(row, id, h)
		if m.typeEmb != nil {
			addInto(row, m.typeEmb[:h]) // token type 0
		}
		if m.posEmb != nil {
			addInto(row, m.posEmb[t*h:t*h+h])
		}
	}
	eps := float32(cfg.LayerNormEps)
	layerNormRows(x, seq, h, m.embNormW, m.embNormB, eps)

	// ── Transformer stack (post-norm) ──
	for li := range m.layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		// CLS pooling reads only row 0 of the final hidden state, so the
		// last layer computes just that query row: keys and values still
		// span the whole sequence, and row 0 is bit-identical to a full-row
		// pass (each output element is the same serial dot product).
		rows := seq
		if li == len(m.layers)-1 && cfg.Pooling == PoolCLS {
			rows = 1
		}
		if err := m.layerForward(ctx, kern, attn, threads, x, seq, rows, &m.layers[li], scratch); err != nil {
			return nil, err
		}
	}

	// ── Pooling ──
	pooled := scratch.pooled
	clear(pooled)
	switch cfg.Pooling {
	case PoolCLS:
		copy(pooled, x[:h])
	case PoolMean:
		for t := 0; t < seq; t++ {
			addInto(pooled, x[t*h:t*h+h])
		}
		inv := 1.0 / float32(seq)
		for i := range pooled {
			pooled[i] *= inv
		}
	}

	// ── Matryoshka truncation ──
	if outDim != 0 && outDim != h {
		if cfg.Matryoshka.LayerNorm {
			// Parameter-free layer norm, F.layer_norm default eps.
			layerNormRows(pooled, 1, h, nil, nil, 1e-5)
		}
		pooled = pooled[:outDim]
	}

	// ── L2 normalization ──
	if cfg.Normalize {
		var ss float64
		for _, v := range pooled {
			ss += float64(v) * float64(v)
		}
		if ss > 0 {
			inv := float32(1.0 / math.Sqrt(ss))
			for i := range pooled {
				pooled[i] *= inv
			}
		}
	}
	out := make([]float32, len(pooled))
	copy(out, pooled)
	return out, nil
}

// forwardScratch holds per-call buffers so concurrent EncodeTokens calls
// never share mutable state. Each forward takes one from the Model's pool
// and returns it when done, so steady-state embedding allocates only the
// output vector. Every buffer is fully overwritten before it is read.
//
// Allocation bound: seq is validated ≤ EffectiveMaxSeqLen before scratch is
// sized (the HIGH-2 trained-context clamp), so the dominant seq² and
// seq×inter buffers are capped — nomic at 2048: ~16MB scores + ~25MB per FFN
// buffer per in-flight forward — and EmbedBatch bounds in-flight forwards to
// its worker-pool size. A declared-8192 forward (~268MB scores alone) is
// unreachable. Pooled buffers are released by the GC like any sync.Pool.
type forwardScratch struct {
	x       []float32     // [seq, h] hidden state
	pooled  []float32     // [h]
	qkv     []float32     // [seq, 3h]
	heads   []headScratch // one per attention worker
	attnCat []float32     // [seq, h]
	attnOut []float32     // [seq, h]
	ffnH    []float32     // [seq, inter]
	ffnGate []float32     // [seq, inter] (swiglu only)
	ffnOut  []float32     // [seq, h]
}

// headScratch is one attention worker's per-head staging.
type headScratch struct {
	q       []float32 // [seq, headDim]
	k       []float32 // [padSeq(seq), headDim]
	vT      []float32 // [headDim, padSeq(seq)]
	scores  []float32 // [seq, padSeq(seq)]
	headOut []float32 // [seq, headDim]
}

// attnPlan says how a forward runs attention: on workers goroutines (1 =
// serial over heads), with kern for the per-head GEMMs.
type attnPlan struct {
	kern    Kernel
	workers int
}

// getScratch returns pooled scratch sized for a seq-token forward with
// headWorkers attention workers.
func (m *Model) getScratch(seq, headWorkers int) *forwardScratch {
	s, _ := m.scratch.Get().(*forwardScratch)
	if s == nil {
		s = &forwardScratch{}
	}
	h, inter, hd := m.cfg.HiddenSize, m.cfg.IntermediateSize, m.headDim
	s.x = grow(s.x, seq*h)
	s.pooled = grow(s.pooled, h)
	s.qkv = grow(s.qkv, seq*3*h)
	kp := padSeq(seq)
	for len(s.heads) < headWorkers {
		s.heads = append(s.heads, headScratch{})
	}
	for i := range s.heads[:headWorkers] {
		hs := &s.heads[i]
		hs.q = grow(hs.q, seq*hd)
		hs.k = grow(hs.k, kp*hd)
		hs.vT = grow(hs.vT, hd*kp)
		hs.scores = grow(hs.scores, seq*kp)
		hs.headOut = grow(hs.headOut, seq*hd)
	}
	s.attnCat = grow(s.attnCat, seq*h)
	s.attnOut = grow(s.attnOut, seq*h)
	s.ffnH = grow(s.ffnH, seq*inter)
	if m.cfg.Activation == ActSwiGLU {
		s.ffnGate = grow(s.ffnGate, seq*inter)
	}
	s.ffnOut = grow(s.ffnOut, seq*h)
	return s
}

// padSeq rounds a sequence length up to the attention GEMMs' k block (8).
func padSeq(seq int) int { return (seq + 7) &^ 7 }

// grow returns buf resliced to n elements, reallocating only when it is too
// small. Contents are unspecified.
func grow(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

// layerForward applies one post-norm transformer block to x in place.
// Keys and values span all seq rows; queries — and so every output — cover
// only the first rows rows (rows < seq lets CLS pooling skip the unread rows
// of the last layer). threads is the budget for the elementwise activation.
// ctx is re-checked between attention and FFN (LOW-3): at trained-context
// sequence lengths a single layer is long enough that per-layer checks alone
// leave cancellation latency in the seconds.
func (m *Model) layerForward(ctx context.Context, kern Kernel, attn attnPlan, threads int, x []float32, seq, rows int, lw *layerWeights, s *forwardScratch) error {
	cfg := m.cfg
	h := cfg.HiddenSize

	// qkv = x·Wqkvᵀ (+bias): [seq, 3h], laid out q(heads-major) | k | v.
	mmT(kern, s.qkv, x, lw.qkvW, seq, 3*h, h, lw.qkvB)

	// Attention. Each head writes its own columns of attnCat, so workers
	// never share an output. Heads are handed out from this goroutine, which
	// keeps the per-head cancellation checkpoint (LOW-3) in one place.
	if attn.workers == 1 {
		for head := 0; head < cfg.NumHeads; head++ {
			if err := ctx.Err(); err != nil {
				return err
			}
			m.attendHead(attn.kern, &s.heads[0], s, head, seq, rows, head == 0)
		}
	} else {
		next := make(chan int)
		var wg sync.WaitGroup
		for w := 0; w < attn.workers; w++ {
			wg.Add(1)
			go func(hs *headScratch) {
				defer wg.Done()
				first := true
				for head := range next {
					m.attendHead(attn.kern, hs, s, head, seq, rows, first)
					first = false
				}
			}(&s.heads[w])
		}
		var err error
		for head := 0; head < cfg.NumHeads; head++ {
			if err = ctx.Err(); err != nil {
				break
			}
			next <- head
		}
		close(next)
		wg.Wait()
		if err != nil {
			return err
		}
	}

	// Output projection, residual, norm1.
	mmT(kern, s.attnOut, s.attnCat, lw.outW, rows, h, h, lw.outB)
	addInto(s.attnOut[:rows*h], x[:rows*h])
	copy(x[:rows*h], s.attnOut)
	layerNormRows(x, rows, h, lw.norm1W, lw.norm1B, float32(cfg.LayerNormEps))

	// Cancellation point between attention and FFN: the FFN GEMMs are the
	// other half of a layer's work.
	if err := ctx.Err(); err != nil {
		return err
	}

	// FFN.
	inter := cfg.IntermediateSize
	switch cfg.Activation {
	case ActGELU:
		mmT(kern, s.ffnH, x, lw.fc1W, rows, inter, h, lw.fc1B)
		parallelChunks(rows*inter, threads, func(lo, hi int) {
			for i, v := range s.ffnH[lo:hi] {
				s.ffnH[lo+i] = geluF32(v)
			}
		})
	case ActSwiGLU:
		// nomic NomciBertGatedMLP: y = fc11(x) * silu(fc12(x)).
		mmT(kern, s.ffnH, x, lw.fc1W, rows, inter, h, lw.fc1B)
		mmT(kern, s.ffnGate, x, lw.gateW, rows, inter, h, lw.gateB)
		parallelChunks(rows*inter, threads, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				g := s.ffnGate[i]
				s.ffnH[i] *= g * sigmoidF32(g)
			}
		})
	}
	mmT(kern, s.ffnOut, s.ffnH, lw.fc2W, rows, h, inter, lw.fc2B)

	// Residual, norm2.
	addInto(s.ffnOut[:rows*h], x[:rows*h])
	copy(x[:rows*h], s.ffnOut)
	layerNormRows(x, rows, h, lw.norm2W, lw.norm2B, float32(cfg.LayerNormEps))
	return nil
}

// attendHead computes one attention head from s.qkv into its columns of
// s.attnCat, staging through hs. first marks the first head this hs serves
// in the layer, which clears the key-axis padding.
//
// The key axis is zero-padded to a multiple of 8 so both attention GEMMs run
// whole SIMD blocks: padded key rows score 0, their probabilities are
// cleared after the softmax, and the matching zero columns of vᵀ add exact
// zeros to the output. No head writes the padding, so clearing it once per
// layer per worker suffices.
func (m *Model) attendHead(kern Kernel, hs *headScratch, s *forwardScratch, head, seq, rows int, first bool) {
	cfg := m.cfg
	h := cfg.HiddenSize
	hd := m.headDim
	half := hd / 2
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	kp := padSeq(seq)
	if first {
		clear(hs.k[seq*hd : kp*hd])
		for d := 0; d < hd; d++ {
			clear(hs.vT[d*kp+seq : d*kp+kp])
		}
	}

	qOff := head * hd
	kOff := h + head*hd
	vOff := 2*h + head*hd
	for t := 0; t < seq; t++ {
		row := s.qkv[t*3*h:]
		if t < rows {
			copy(hs.q[t*hd:t*hd+hd], row[qOff:qOff+hd])
		}
		copy(hs.k[t*hd:t*hd+hd], row[kOff:kOff+hd])
		for d, v := range row[vOff : vOff+hd] {
			hs.vT[d*kp+t] = v
		}
	}

	if cfg.PositionEncoding == PosRoPE {
		// NeoX-style half rotation over the full head dim, matching
		// nomic's apply_rotary_emb (non-interleaved).
		for t := 0; t < seq; t++ {
			cosRow := m.ropeCos[t*half : t*half+half]
			sinRow := m.ropeSin[t*half : t*half+half]
			if t < rows {
				rotateHalf(hs.q[t*hd:t*hd+hd], cosRow, sinRow, half)
			}
			rotateHalf(hs.k[t*hd:t*hd+hd], cosRow, sinRow, half)
		}
	}

	// scores = q·kᵀ / √d, softmax rows.
	kern.MatMulT(hs.scores, hs.q, hs.k, rows, kp, hd, nil)
	for t := 0; t < rows; t++ {
		row := hs.scores[t*kp : t*kp+kp]
		softmaxScaled(row[:seq], scale)
		clear(row[seq:])
	}

	// headOut = probs·v via probs·(vᵀ)ᵀ.
	kern.MatMulT(hs.headOut, hs.scores, hs.vT, rows, hd, kp, nil)

	for t := 0; t < rows; t++ {
		copy(s.attnCat[t*h+head*hd:t*h+head*hd+hd], hs.headOut[t*hd:t*hd+hd])
	}
}

// rotateHalf applies the NeoX rotation in place:
// out[i] = x[i]·cos_i − x[i+half]·sin_i; out[i+half] = x[i+half]·cos_i + x[i]·sin_i.
func rotateHalf(x, cos, sin []float32, half int) {
	for i := 0; i < half; i++ {
		x1, x2 := x[i], x[i+half]
		c, s := cos[i], sin[i]
		x[i] = x1*c - x2*s
		x[i+half] = x2*c + x1*s
	}
}

// softmaxScaled computes softmax(row·scale) in place with max subtraction.
// scale > 0, so max(row)·scale is the max of the scaled row. The
// denominator accumulates in float64 over a fixed order.
func softmaxScaled(row []float32, scale float32) {
	maxV := row[0]
	for _, v := range row[1:] {
		if v > maxV {
			maxV = v
		}
	}
	maxV *= scale
	var sum float64
	for i, v := range row {
		d := v*scale - maxV
		if d < expClampLo {
			d = expClampLo
		}
		e := expF32(d)
		row[i] = e
		sum += float64(e)
	}
	inv := float32(1.0 / sum)
	for i := range row {
		row[i] *= inv
	}
}

// layerNormRows normalizes each h-wide row of x in place. weight/bias may be
// nil for a parameter-free norm (the matryoshka recipe).
func layerNormRows(x []float32, rows, h int, weight, bias []float32, eps float32) {
	for r := 0; r < rows; r++ {
		row := x[r*h : r*h+h]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(h)
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(h)
		inv := float32(1.0 / math.Sqrt(variance+float64(eps)))
		mf := float32(mean)
		if weight != nil {
			for i := range row {
				row[i] = (row[i]-mf)*inv*weight[i] + bias[i]
			}
		} else {
			for i := range row {
				row[i] = (row[i] - mf) * inv
			}
		}
	}
}

func addInto(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}
