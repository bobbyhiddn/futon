package native

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Toy-model gates. The references were produced by the real upstream
// implementations (transformers BertModel; the pinned nomic-bert modeling
// code) on random weights, so agreement here proves the ARCHITECTURE, not
// memorized numbers. Toys are tiny (32-dim) so f32-order divergence is
// visible; 1e-5 cosine slack and 2e-4 absolute slack are far above f32 noise
// and far below any architectural mistake.
const (
	toyCosineMin = 0.99999
	toyAbsTol    = 2e-4
)

type toyExpected struct {
	Packs map[string]struct {
		Pooling      string `json:"pooling"`
		ForwardCases []struct {
			IDs    []int     `json:"ids"`
			Vector []float32 `json:"vector"`
		} `json:"forward_cases"`
		TextCases []struct {
			Text   string    `json:"text"`
			IDs    []int     `json:"ids"`
			Vector []float32 `json:"vector"`
		} `json:"text_cases"`
		MatryoshkaCases []struct {
			IDs    []int     `json:"ids"`
			Dim    int       `json:"dim"`
			Vector []float32 `json:"vector"`
		} `json:"matryoshka_cases"`
	} `json:"packs"`
}

func loadToy(t *testing.T, pack string) *Model {
	t.Helper()
	m, err := LoadModel(filepath.Join("testdata", "toy", pack), 0)
	if err != nil {
		t.Fatalf("load toy pack %s: %v", pack, err)
	}
	return m
}

func assertClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: dims %d vs %d", name, len(got), len(want))
	}
	if c := cosine(got, want); c < toyCosineMin {
		t.Errorf("%s: cosine %.7f < %.5f", name, c, toyCosineMin)
	}
	for i := range got {
		if diff := math.Abs(float64(got[i] - want[i])); diff > toyAbsTol {
			t.Errorf("%s: element %d differs by %g (got %v want %v)", name, i, diff, got[i], want[i])
			return
		}
	}
}

// TestToyForward validates the full forward pass for BOTH option sets
// (absolute+GELU+CLS and RoPE+SwiGLU+mean) against upstream-generated
// references — the CI-runnable architecture test that needs no big weights.
func TestToyForward(t *testing.T) {
	var exp toyExpected
	loadJSON(t, filepath.Join("testdata", "toy", "expected.json"), &exp)
	for pack, pexp := range exp.Packs {
		t.Run(pack, func(t *testing.T) {
			m := loadToy(t, pack)
			for i, c := range pexp.ForwardCases {
				got, err := m.EncodeTokens(context.Background(), c.IDs, 0)
				if err != nil {
					t.Fatalf("case %d: %v", i, err)
				}
				assertClose(t, pack, got, c.Vector)
			}
			for _, c := range pexp.MatryoshkaCases {
				got, err := m.EncodeTokens(context.Background(), c.IDs, c.Dim)
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != c.Dim {
					t.Fatalf("matryoshka dim %d, want %d", len(got), c.Dim)
				}
				assertClose(t, pack+"/matryoshka", got, c.Vector)
			}
		})
	}
}

// TestToyProviderEndToEnd runs text → tokenizer → forward through the
// Provider, against expectations computed with the HF tokenizer + upstream
// model. Includes the empty string and an [UNK]-bearing text.
func TestToyProviderEndToEnd(t *testing.T) {
	var exp toyExpected
	loadJSON(t, filepath.Join("testdata", "toy", "expected.json"), &exp)
	pexp := exp.Packs["toy-absolute-gelu"]
	if len(pexp.TextCases) == 0 {
		t.Fatal("no text cases")
	}
	p, err := New(Options{PackDir: filepath.Join("testdata", "toy", "toy-absolute-gelu")})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	for _, c := range pexp.TextCases {
		got, err := p.Embed(context.Background(), c.Text)
		if err != nil {
			t.Fatalf("embed %q: %v", c.Text, err)
		}
		assertClose(t, "text:"+c.Text, got, c.Vector)
	}
}

// TestDeterminism pins the within-host determinism contract: repeated runs
// and different kernel thread counts produce bit-identical vectors.
// (Cross-host bit-equality is NOT claimed — libm/FMA differ across
// platforms; cosine parity gates cover that.)
func TestDeterminism(t *testing.T) {
	m := loadToy(t, "toy-rope-swiglu")
	ids := []int{2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3}

	ref, err := m.EncodeTokens(context.Background(), ids, 0)
	if err != nil {
		t.Fatal(err)
	}
	for run := 0; run < 3; run++ {
		got, err := m.EncodeTokens(context.Background(), ids, 0)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("run %d: element %d differs bitwise", run, i)
			}
		}
	}
	for _, threads := range []int{1, 2, runtime.NumCPU()} {
		m.SetKernelThreads(threads)
		got, err := m.EncodeTokens(context.Background(), ids, 0)
		if err != nil {
			t.Fatal(err)
		}
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("threads=%d: element %d differs bitwise", threads, i)
			}
		}
	}
}

// TestBatchMatchesSingle pins the no-padding design: EmbedBatch must be
// bit-identical to per-text Embed. This is the guard against the classic
// silent-recall-degradation bug where padding leaks into attention or
// pooling — here it cannot, and this test keeps it that way if batching is
// ever rewritten to use real padded batches.
func TestBatchMatchesSingle(t *testing.T) {
	p, err := New(Options{PackDir: filepath.Join("testdata", "toy", "toy-rope-swiglu")})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	texts := []string{
		"the cat sat on the mat.",
		"hello",
		"",
		"snowman runs",
		"the the the the the the the the",
		"unaffable!",
	}
	batch, err := p.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range texts {
		single, err := p.Embed(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		if len(single) != len(batch[i]) {
			t.Fatalf("text %d: dim mismatch", i)
		}
		for j := range single {
			if single[j] != batch[i][j] {
				t.Fatalf("text %d (%q): batch and single differ bitwise at %d", i, text, j)
			}
		}
	}
}

// cancelAfterN reports Canceled from the (n+1)-th Err() call onward —
// a deterministic way to land cancellation at every ctx checkpoint in the
// forward pass, including the LOW-3 checkpoints INSIDE a layer (per
// attention head, and between attention and FFN).
type cancelAfterN struct {
	context.Context
	n int
}

func (c *cancelAfterN) Err() error {
	if c.n <= 0 {
		return context.Canceled
	}
	c.n--
	return nil
}

// TestCancellationInsideLayer pins LOW-3: cancellation must abort the
// forward pass at intra-layer checkpoints, not only at layer boundaries.
func TestCancellationInsideLayer(t *testing.T) {
	m := loadToy(t, "toy-rope-swiglu")
	ids := []int{2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 3}

	// Count how many checkpoints one full run passes.
	probe := &cancelAfterN{Context: context.Background(), n: 1 << 30}
	if _, err := m.EncodeTokens(probe, ids, 0); err != nil {
		t.Fatal(err)
	}
	total := 1<<30 - probe.n
	// 2 layers × (1 layer-boundary + 4 heads + 1 attention/FFN boundary):
	// intra-layer checkpoints must exist beyond the per-layer ones.
	if total < 2*(1+4+1) {
		t.Fatalf("only %d ctx checkpoints in a 2-layer forward; intra-layer checks missing", total)
	}
	// Landing the cancellation on every checkpoint must abort with
	// context.Canceled and no vector.
	for n := 0; n < total; n++ {
		vec, err := m.EncodeTokens(&cancelAfterN{Context: context.Background(), n: n}, ids, 0)
		if err != context.Canceled {
			t.Fatalf("checkpoint %d: err = %v, want context.Canceled", n, err)
		}
		if vec != nil {
			t.Fatalf("checkpoint %d: got a vector despite cancellation", n)
		}
	}
}

// TestTrainedContextClampToy pins the HIGH-2 clamp in CI without real
// weights: when a pack declares max_trained_positions below max_seq_len, the
// provider must truncate to the TRAINED context (same head-truncation
// semantics, [SEP] preserved) and the model must refuse longer id sequences
// outright — plain RoPE extrapolation past training is a silent-degradation
// regime and is never run.
func TestTrainedContextClampToy(t *testing.T) {
	dir := copyPack(t, filepath.Join("testdata", "toy", "toy-rope-swiglu"))
	mutateJSON(t, filepath.Join(dir, "config.json"), func(m map[string]interface{}) {
		m["max_trained_positions"] = float64(16)
	})
	rehashManifest(t, dir)
	p, err := New(Options{PackDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if got := p.model.Config().EffectiveMaxSeqLen(); got != 16 {
		t.Fatalf("EffectiveMaxSeqLen = %d, want 16", got)
	}

	long := "the cat sat on the mat . hello world snow man runs . the cat sat on the mat . hello world"
	// The input must genuinely exceed the trained context for the case to
	// mean anything.
	if ids := p.tok.Encode(long, p.model.Config().MaxSeqLen); len(ids) <= 16 {
		t.Fatalf("test input only %d tokens; must exceed the trained context", len(ids))
	}
	got, err := p.Embed(context.Background(), long)
	if err != nil {
		t.Fatal(err)
	}
	idsClamped := p.tok.Encode(long, 16)
	if len(idsClamped) != 16 {
		t.Fatalf("clamped encode produced %d ids, want exactly 16", len(idsClamped))
	}
	want, err := p.model.EncodeTokens(context.Background(), idsClamped, p.dims)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("clamped embed differs bitwise at %d from explicit 16-token truncation", i)
		}
	}

	// Direct callers cannot sneak past the clamp either: 17 ids within the
	// DECLARED max (64) but past the trained context must be refused loudly.
	over := make([]int, 17)
	for i := range over {
		over[i] = 5
	}
	if _, err := p.model.EncodeTokens(context.Background(), over, 0); err == nil {
		t.Fatal("sequence past the trained context was accepted — RoPE extrapolation regime reachable")
	}
}

// TestBatchPlan pins MED-1's resource sizing: a bounded worker pool (never
// goroutine-per-text) whose per-forward kernels split the thread budget
// instead of multiplying it.
func TestBatchPlan(t *testing.T) {
	cases := []struct {
		nTexts, batchWorkers, threads int
		wantWorkers, wantPerForward   int
	}{
		{1, 4, 4, 1, 4},   // single text: one worker, full budget
		{2, 4, 4, 2, 2},   // split evenly
		{100, 4, 4, 4, 1}, // big batch: workers capped, one thread each
		{100, 8, 4, 8, 1}, // never below 1 thread per forward
		{3, 4, 4, 3, 1},   // workers capped at batch size
		{0, 4, 4, 1, 4},   // degenerate: still a valid plan
		{100, 1, 4, 1, 4}, // serial batch keeps the full budget
		{100, 16, 16, 16, 1},
	}
	for _, c := range cases {
		w, pf := batchPlan(c.nTexts, c.batchWorkers, c.threads)
		if w != c.wantWorkers || pf != c.wantPerForward {
			t.Errorf("batchPlan(%d,%d,%d) = (%d,%d), want (%d,%d)",
				c.nTexts, c.batchWorkers, c.threads, w, pf, c.wantWorkers, c.wantPerForward)
		}
		if w*pf > c.threads && pf != 1 {
			t.Errorf("batchPlan(%d,%d,%d) oversubscribes: %d workers × %d threads",
				c.nTexts, c.batchWorkers, c.threads, w, pf)
		}
	}
}

// TestBatchMatchesSingleAcrossWorkerCounts extends the batch ≡ single
// bit-equality pin across the MED-1 knobs: any worker-pool size and any
// per-forward thread split must produce identical bits (the kernel
// determinism contract makes the split invisible).
func TestBatchMatchesSingleAcrossWorkerCounts(t *testing.T) {
	p, err := New(Options{PackDir: filepath.Join("testdata", "toy", "toy-rope-swiglu")})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	texts := []string{
		"the cat sat on the mat.", "hello", "", "snowman runs",
		"the the the the the the the the", "unaffable!", "a b c d e f g",
	}
	singles := make([][]float32, len(texts))
	for i, text := range texts {
		if singles[i], err = p.Embed(context.Background(), text); err != nil {
			t.Fatal(err)
		}
	}
	for _, workers := range []int{1, 2, 3, 16} {
		for _, threads := range []int{1, 2, runtime.NumCPU()} {
			p.batchWorkers, p.threads = workers, threads
			batch, err := p.EmbedBatch(context.Background(), texts)
			if err != nil {
				t.Fatal(err)
			}
			for i := range texts {
				for j := range singles[i] {
					if batch[i][j] != singles[i][j] {
						t.Fatalf("workers=%d threads=%d text %d: differs bitwise at %d",
							workers, threads, i, j)
					}
				}
			}
		}
	}
}

// TestBatchGoroutinesBounded pins that EmbedBatch runs a worker POOL: a
// large batch must not spawn a goroutine per text (the pre-fix code spawned
// len(texts) goroutines gated by a semaphore).
func TestBatchGoroutinesBounded(t *testing.T) {
	p, err := New(Options{PackDir: filepath.Join("testdata", "toy", "toy-rope-swiglu")})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	texts := make([]string, 256)
	for i := range texts {
		texts[i] = "the cat sat on the mat . hello world snow man runs"
	}
	base := runtime.NumGoroutine()

	peak := base
	done := make(chan struct{})
	sampled := make(chan struct{})
	go func() {
		defer close(sampled)
		for {
			select {
			case <-done:
				return
			default:
				if n := runtime.NumGoroutine(); n > peak {
					peak = n
				}
			}
		}
	}()
	if _, err := p.EmbedBatch(context.Background(), texts); err != nil {
		t.Fatal(err)
	}
	close(done)
	<-sampled

	// Budget: pool workers + per-forward kernel goroutines + sampler + slack.
	// Goroutine-per-text would put the peak near base+256.
	limit := base + p.batchWorkers*2 + 16
	if peak > limit {
		t.Fatalf("goroutine peak %d exceeds bound %d (base %d) — batch is not a bounded pool", peak, limit, base)
	}
}

func TestModelRejects(t *testing.T) {
	m := loadToy(t, "toy-absolute-gelu")
	ctx := context.Background()
	if _, err := m.EncodeTokens(ctx, nil, 0); err == nil {
		t.Error("empty ids accepted")
	}
	if _, err := m.EncodeTokens(ctx, []int{2, 999, 3}, 0); err == nil {
		t.Error("out-of-vocab id accepted")
	}
	long := make([]int, m.Config().MaxSeqLen+1)
	if _, err := m.EncodeTokens(ctx, long, 0); err == nil {
		t.Error("over-long sequence accepted")
	}
	// toy-absolute-gelu has no matryoshka support.
	if _, err := m.EncodeTokens(ctx, []int{2, 3}, 16); err == nil {
		t.Error("matryoshka truncation accepted by non-matryoshka pack")
	}
	if _, err := m.EncodeTokens(ctx, []int{2, 3}, m.Config().HiddenSize+1); err == nil {
		t.Error("dim > hidden accepted")
	}
}

func TestProviderRejects(t *testing.T) {
	if _, err := New(Options{}); err == nil {
		t.Error("empty PackDir accepted")
	}
	if _, err := New(Options{PackDir: filepath.Join("testdata", "toy", "toy-absolute-gelu"), Dimensions: 16}); err == nil {
		t.Error("matryoshka dims accepted for non-matryoshka pack")
	}
	if _, err := New(Options{PackDir: "testdata/does-not-exist"}); err == nil {
		t.Error("missing pack accepted")
	}
	p, err := New(Options{PackDir: filepath.Join("testdata", "toy", "toy-rope-swiglu"), Dimensions: 16})
	if err != nil {
		t.Fatalf("matryoshka dims rejected by matryoshka pack: %v", err)
	}
	if p.Dimensions() != 16 {
		t.Errorf("Dimensions() = %d, want 16", p.Dimensions())
	}
	vec, err := p.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != 16 {
		t.Errorf("embedded dim %d, want 16", len(vec))
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := p.Embed(context.Background(), "hello"); err == nil {
		t.Error("embed after Close accepted")
	}
	if p.IsAvailable(context.Background()) {
		t.Error("IsAvailable true after Close")
	}
}

// TestPackTamperFailsLoudly corrupts a pack and expects loud load failures —
// the "malformed files fail loudly" contract end to end.
//
// Config-mutating subtests call rehashManifest so they exercise their
// ORIGINAL rejection path (the manifest sha256 gate would otherwise fire
// first and mask a regression in the deeper check).
func TestPackTamperFailsLoudly(t *testing.T) {
	src := filepath.Join("testdata", "toy", "toy-absolute-gelu")

	t.Run("extra_tensor", func(t *testing.T) {
		dir := copyPack(t, src)
		// Append a rogue tensor by rewriting config to declare fewer layers:
		// the loader must reject leftover tensors.
		mutateJSON(t, filepath.Join(dir, "config.json"), func(m map[string]interface{}) {
			m["num_layers"] = float64(1)
		})
		rehashManifest(t, dir)
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("pack with unexpected tensors loaded")
		}
	})
	t.Run("wrong_format", func(t *testing.T) {
		dir := copyPack(t, src)
		mutateJSON(t, filepath.Join(dir, "config.json"), func(m map[string]interface{}) {
			m["pack_format"] = float64(99)
		})
		rehashManifest(t, dir)
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("unknown pack_format loaded")
		}
	})
	t.Run("prenorm_rejected", func(t *testing.T) {
		dir := copyPack(t, src)
		mutateJSON(t, filepath.Join(dir, "config.json"), func(m map[string]interface{}) {
			m["pre_norm"] = true
		})
		rehashManifest(t, dir)
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("pre_norm pack loaded despite being unimplemented")
		}
	})
	t.Run("truncated_weights", func(t *testing.T) {
		dir := copyPack(t, src)
		path := filepath.Join(dir, "model.safetensors")
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, raw[:len(raw)-100], 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("truncated weights loaded")
		}
	})

	// ── LOW-2: load-time manifest verification ──
	t.Run("missing_manifest", func(t *testing.T) {
		dir := copyPack(t, src)
		if err := os.Remove(filepath.Join(dir, "manifest.json")); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("pack without manifest.json loaded — unverifiable packs must be refused")
		}
	})
	t.Run("edited_config_stale_manifest", func(t *testing.T) {
		dir := copyPack(t, src)
		// Any member edit without a matching manifest must be loud.
		mutateJSON(t, filepath.Join(dir, "config.json"), func(m map[string]interface{}) {
			m["normalize"] = false
		})
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("member/manifest sha256 mismatch loaded")
		}
	})
	t.Run("removed_optional_tensor", func(t *testing.T) {
		dir := copyPack(t, src)
		dropTensor(t, dir, "layers.0.attn.qkv.bias")
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("weights with a deleted optional bias loaded (sha256 gate missed it)")
		}
	})
	t.Run("removed_optional_tensor_rehashed_manifest", func(t *testing.T) {
		// The deeper attack: delete the optional tensor AND fix up the
		// manifest hashes. The recorded tensor-name set must still refuse —
		// a missing optional bias is a silently different function.
		dir := copyPack(t, src)
		dropTensor(t, dir, "layers.0.attn.qkv.bias")
		rehashManifest(t, dir)
		if _, err := LoadModel(dir, 1); err == nil {
			t.Fatal("weights with a deleted optional bias loaded despite tensor_names record")
		}
	})
}

// rehashManifest recomputes manifest.json member hashes/sizes after a test
// deliberately mutates pack files (simulating an attacker or a rebuilt
// manifest). It does NOT touch tensor_names — that record is the point.
func rehashManifest(t *testing.T, dir string) {
	t.Helper()
	mutateJSON(t, filepath.Join(dir, "manifest.json"), func(m map[string]interface{}) {
		members, ok := m["members"].(map[string]interface{})
		if !ok {
			t.Fatal("manifest has no members")
		}
		for name := range members {
			path := filepath.Join(dir, name)
			sum, err := sha256File(path)
			if err != nil {
				t.Fatal(err)
			}
			info, err := os.Stat(path)
			if err != nil {
				t.Fatal(err)
			}
			members[name] = map[string]interface{}{"sha256": sum, "size": info.Size()}
		}
	})
}

// dropTensor rewrites <dir>/model.safetensors without the named tensor.
func dropTensor(t *testing.T, dir, name string) {
	t.Helper()
	path := filepath.Join(dir, "model.safetensors")
	tensors, err := LoadSafetensors(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tensors[name]; !ok {
		t.Fatalf("tensor %q not in pack", name)
	}
	delete(tensors, name)
	entries := make(map[string]stTestEntry, len(tensors))
	for n, tensor := range tensors {
		entries[n] = stTestEntry{dtype: "F32", shape: tensor.Shape, data: f32Bytes(tensor.Data...)}
	}
	if err := os.WriteFile(path, buildSafetensors(t, entries), 0o600); err != nil {
		t.Fatal(err)
	}
}

func copyPack(t *testing.T, src string) string {
	t.Helper()
	dst := t.TempDir()
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		raw, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), raw, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

func mutateJSON(t *testing.T, path string, fn func(map[string]interface{})) {
	t.Helper()
	var m map[string]interface{}
	loadJSON(t, path, &m)
	fn(m)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
}
