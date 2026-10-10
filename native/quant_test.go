package native

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// ── Q8 quantization primitives ──────────────────────────────────────────────

func TestQ8QuantizeRow(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	src := randSlice(rng, 257)
	dst := make([]int8, len(src))
	scale := quantizeRowQ8(dst, src)
	if scale <= 0 {
		t.Fatalf("scale = %v for a non-zero row", scale)
	}
	var amax float64
	for _, v := range src {
		if a := math.Abs(float64(v)); a > amax {
			amax = a
		}
	}
	for i, q := range dst {
		if q == -128 {
			t.Fatalf("element %d quantized to -128; scheme must be symmetric", i)
		}
		// Round-trip error bound: half a quantization step.
		if err := math.Abs(float64(src[i]) - float64(q)*float64(scale)); err > float64(scale)/2+1e-7 {
			t.Fatalf("element %d round-trip error %v exceeds scale/2 (%v)", i, err, scale/2)
		}
	}
	if got := float64(scale); math.Abs(got-amax/127) > 1e-9*amax {
		t.Fatalf("scale %v, want absmax/127 = %v", got, amax/127)
	}

	// All-zero row: scale 0, all-zero codes.
	zeros := make([]float32, 8)
	scale = quantizeRowQ8(dst[:8], zeros)
	if scale != 0 {
		t.Fatalf("zero row scale = %v, want 0", scale)
	}
	for _, q := range dst[:8] {
		if q != 0 {
			t.Fatal("zero row produced non-zero codes")
		}
	}
}

// TestQ8MatMulTCorrectness checks every Q8-capable kernel against an exact
// int64/float64 reference over the SAME quantized operands — this pins the
// integer accumulation and rescale, independent of quantization loss.
func TestQ8MatMulTCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {2, 3, 7}, {4, 4, 8}, {5, 9, 16}, {7, 16, 33}, {13, 29, 65},
	}
	for _, sh := range shapes {
		a := randSlice(rng, sh.m*sh.k)
		bw := weightMat{f32: randSlice(rng, sh.n*sh.k)}
		bias := randSlice(rng, sh.n)
		q := quantizeQ8(bw, sh.n, sh.k)

		// Reference: quantize a the same way, exact integer dots in int64.
		qa := make([]int8, sh.m*sh.k)
		aScale := make([]float32, sh.m)
		for i := 0; i < sh.m; i++ {
			aScale[i] = quantizeRowQ8(qa[i*sh.k:(i+1)*sh.k], a[i*sh.k:(i+1)*sh.k])
		}
		want := make([]float64, sh.m*sh.n)
		for i := 0; i < sh.m; i++ {
			for j := 0; j < sh.n; j++ {
				var sum int64
				for p := 0; p < sh.k; p++ {
					sum += int64(qa[i*sh.k+p]) * int64(q.data[j*sh.k+p])
				}
				want[i*sh.n+j] = float64(float32(sum)*(aScale[i]*q.scale[j])) + float64(bias[j])
			}
		}

		for _, threads := range []int{1, 3, 8} {
			kern, ok := newKernel(threads).(q8Kernel)
			if !ok {
				t.Fatal("shipped kernel does not implement q8Kernel")
			}
			got := make([]float32, sh.m*sh.n)
			kern.MatMulTQ8(got, a, q.data, q.scale, sh.m, sh.n, sh.k, bias)
			for i := range want {
				if diff := math.Abs(float64(got[i]) - want[i]); diff > 1e-4 {
					t.Fatalf("shape %+v threads=%d: element %d = %v, want %v",
						sh, threads, i, got[i], want[i])
				}
			}
		}
	}
}

// TestQ8DeterministicAcrossThreads extends the bit-determinism contract to
// the Q8 path: thread count never changes any output bit.
func TestQ8DeterministicAcrossThreads(t *testing.T) {
	rng := rand.New(rand.NewSource(29))
	m, n, k := 37, 53, 129
	a := randSlice(rng, m*k)
	q := quantizeQ8(weightMat{f32: randSlice(rng, n*k)}, n, k)
	bias := randSlice(rng, n)

	ref := make([]float32, m*n)
	newKernel(1).(q8Kernel).MatMulTQ8(ref, a, q.data, q.scale, m, n, k, bias)
	for _, threads := range []int{2, 4, 16} {
		got := make([]float32, m*n)
		newKernel(threads).(q8Kernel).MatMulTQ8(got, a, q.data, q.scale, m, n, k, bias)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("threads=%d: element %d differs bitwise", threads, i)
			}
		}
	}
}

// ── Wiring: off by default, on only under the env flag ──────────────────────

func layerWeightMats(m *Model) []*weightMat {
	var mats []*weightMat
	for i := range m.layers {
		lw := &m.layers[i]
		mats = append(mats, &lw.qkvW, &lw.outW, &lw.fc1W, &lw.fc2W)
		if lw.gateW.f16 != nil || lw.gateW.f32 != nil || lw.gateW.q8 != nil {
			mats = append(mats, &lw.gateW)
		}
	}
	return mats
}

func TestQ8OffByDefault(t *testing.T) {
	t.Setenv(envNativeQ8, "")
	m := loadToy(t, "toy-rope-swiglu")
	for _, w := range layerWeightMats(m) {
		if w.q8 != nil {
			t.Fatal("Q8 weights present without FUTON_NATIVE_Q8=1 — the experimental path leaked into a default load")
		}
	}
}

// TestQ8ToyForward proves the end-to-end wiring: with the flag set, every
// layer GEMM weight is quantized and the forward pass still tracks the
// upstream toy references closely (int8 noise on a 32-dim toy is far larger
// than on real 768-dim packs; the real evidence is TestQ8ParityRealWeights).
func TestQ8ToyForward(t *testing.T) {
	t.Setenv(envNativeQ8, "1")
	var exp toyExpected
	loadJSON(t, filepath.Join("testdata", "toy", "expected.json"), &exp)
	for pack, pexp := range exp.Packs {
		t.Run(pack, func(t *testing.T) {
			m := loadToy(t, pack)
			for _, w := range layerWeightMats(m) {
				if w.q8 == nil || w.f16 != nil || w.f32 != nil {
					t.Fatal("FUTON_NATIVE_Q8=1 load kept float layer weights")
				}
			}
			for i, c := range pexp.ForwardCases {
				got, err := m.EncodeTokens(context.Background(), c.IDs, 0)
				if err != nil {
					t.Fatalf("case %d: %v", i, err)
				}
				if c := cosine(got, c.Vector); c < 0.98 {
					t.Errorf("%s case %d: cosine %.6f < 0.98 — Q8 wiring is broken, not just noisy", pack, i, c)
				}
			}
		})
	}
}

// TestQ8ParityRealWeights is the Phase-5 evidence run for the pre-agreed
// decision rule (Q8 ships as an option only if parity min ≥ 0.999). It is
// opt-in — FUTON_NATIVE_Q8_EVAL=1 — so normal suites never pay for it:
//
//	FUTON_NATIVE_Q8_EVAL=1 go test ./internal/embed/native -run TestQ8Parity -v
//
// The PARITY SUMMARY lines are the evidence table's min/mean columns.
func TestQ8ParityRealWeights(t *testing.T) {
	if os.Getenv("FUTON_NATIVE_Q8_EVAL") != "1" {
		t.Skip("Q8 parity evaluation is opt-in: set FUTON_NATIVE_Q8_EVAL=1")
	}
	t.Setenv(envNativeQ8, "1")
	for _, modelID := range []string{PackNomicEmbedTextV15, PackBGESmallENV15, PackArcticEmbedMLong} {
		t.Run(modelID, func(t *testing.T) {
			runParity(t, modelID)
		})
	}
}
