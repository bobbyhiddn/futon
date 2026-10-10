package native

import (
	"math"
	"math/rand"
	"testing"
)

// naiveMatMulT is the obviously-correct float64 reference.
func naiveMatMulT(a, b []float32, m, n, k int, bias []float32) []float32 {
	out := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*k+p]) * float64(b[j*k+p])
			}
			if bias != nil {
				s += float64(bias[j])
			}
			out[i*n+j] = float32(s)
		}
	}
	return out
}

func randSlice(rng *rand.Rand, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(rng.NormFloat64())
	}
	return out
}

func TestKernelMatMulTCorrectness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {1, 7, 5}, {3, 4, 9}, {4, 4, 4}, {5, 3, 8},
		{7, 16, 33}, {8, 8, 64}, {13, 29, 17}, {32, 6, 100},
	}
	for _, sh := range shapes {
		for _, withBias := range []bool{false, true} {
			a := randSlice(rng, sh.m*sh.k)
			b := randSlice(rng, sh.n*sh.k)
			var bias []float32
			if withBias {
				bias = randSlice(rng, sh.n)
			}
			want := naiveMatMulT(a, b, sh.m, sh.n, sh.k, bias)
			for _, threads := range []int{1, 3, 8} {
				kern := newKernel(threads)
				got := make([]float32, sh.m*sh.n)
				kern.MatMulT(got, a, b, sh.m, sh.n, sh.k, bias)
				for i := range want {
					if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 {
						t.Fatalf("shape %+v bias=%v threads=%d: element %d = %v, want %v",
							sh, withBias, threads, i, got[i], want[i])
					}
				}
			}
		}
	}
}

// TestKernelDeterministicAcrossThreads pins the bit-determinism contract:
// thread count decides which goroutine computes an output element, never the
// reduction order within it, so results are bit-identical for any thread
// count. Assembly kernels slotted in later must preserve this property.
func TestKernelDeterministicAcrossThreads(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	m, n, k := 37, 53, 129
	a := randSlice(rng, m*k)
	b := randSlice(rng, n*k)
	bias := randSlice(rng, n)

	ref := make([]float32, m*n)
	newKernel(1).MatMulT(ref, a, b, m, n, k, bias)
	for _, threads := range []int{2, 4, 8, 16} {
		got := make([]float32, m*n)
		newKernel(threads).MatMulT(got, a, b, m, n, k, bias)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("threads=%d: element %d differs bitwise: %b vs %b",
					threads, i, math.Float32bits(got[i]), math.Float32bits(ref[i]))
			}
		}
	}
}

// TestPureGoEscapeHatch pins FUTON_NATIVE_PUREGO=1: it must force the
// portable scalar kernel on every architecture — the escape hatch
// if a SIMD kernel ever misbehaves in the field, and the benchmark baseline.
func TestPureGoEscapeHatch(t *testing.T) {
	t.Setenv("FUTON_NATIVE_PUREGO", "1")
	if _, ok := newKernel(2).(*goKernel); !ok {
		t.Fatalf("FUTON_NATIVE_PUREGO=1 selected %T, want *goKernel", newKernel(2))
	}
}

func TestKernelPanicsOnShortBuffers(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on short buffer")
		}
	}()
	newKernel(1).MatMulT(make([]float32, 3), make([]float32, 4), make([]float32, 4), 2, 2, 2, nil)
}

// f16Bits converts float32 values to binary16 bits (round-to-nearest-even),
// reference implementation for tests.
func f16Bits(vals []float32) []uint16 {
	out := make([]uint16, len(vals))
	for i, v := range vals {
		out[i] = f32ToF16(v)
	}
	return out
}

// TestKernelMatMulTF16 checks the fp16-weight path of every kernel against
// the float64 reference computed over the dequantized matrix — the two must
// agree because F16→F32 conversion is exact.
func TestKernelMatMulTF16(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	shapes := []struct{ m, n, k int }{
		{1, 1, 1}, {2, 3, 7}, {4, 4, 8}, {5, 9, 16}, {7, 16, 33}, {13, 29, 65}, {6, 2, 100},
	}
	for _, sh := range shapes {
		a := randSlice(rng, sh.m*sh.k)
		bf := randSlice(rng, sh.n*sh.k)
		b16 := f16Bits(bf)
		// Dequantized reference input: exactly what the kernel sees.
		bdeq := make([]float32, len(b16))
		for i, h := range b16 {
			bdeq[i] = f16ToF32(h)
		}
		bias := randSlice(rng, sh.n)
		want := naiveMatMulT(a, bdeq, sh.m, sh.n, sh.k, bias)
		for _, kern := range testKernels(t) {
			got := make([]float32, sh.m*sh.n)
			kern.k.MatMulTF16(got, a, b16, sh.m, sh.n, sh.k, bias)
			for i := range want {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 {
					t.Fatalf("%s shape %+v: element %d = %v, want %v", kern.name, sh, i, got[i], want[i])
				}
			}
		}
	}
}

type namedKernel struct {
	name string
	k    Kernel
}

// testKernels returns every kernel implementation available on this machine.
func testKernels(t *testing.T) []namedKernel {
	t.Helper()
	kerns := []namedKernel{
		{"go/1", &goKernel{threads: 1}},
		{"go/4", &goKernel{threads: 4}},
	}
	if ak := newArchKernel(1); ak != nil {
		kerns = append(kerns, namedKernel{"arch/1", ak}, namedKernel{"arch/3", newArchKernel(3)})
	} else {
		t.Log("no arch kernel on this machine; scalar only")
	}
	return kerns
}

// TestArchKernelMatchesScalar cross-checks the SIMD kernel against the
// portable one within float tolerance (summation order differs, bits may).
func TestArchKernelMatchesScalar(t *testing.T) {
	ak := newArchKernel(2)
	if ak == nil {
		t.Skip("no arch kernel on this machine")
	}
	rng := rand.New(rand.NewSource(23))
	m, n, k := 33, 47, 129
	a := randSlice(rng, m*k)
	b := randSlice(rng, n*k)
	bias := randSlice(rng, n)
	want := make([]float32, m*n)
	(&goKernel{threads: 1}).MatMulT(want, a, b, m, n, k, bias)
	got := make([]float32, m*n)
	ak.MatMulT(got, a, b, m, n, k, bias)
	for i := range want {
		if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-4 {
			t.Fatalf("element %d: arch %v vs scalar %v", i, got[i], want[i])
		}
	}
}

// TestArchKernelDeterministicAcrossThreads extends the determinism contract
// to the SIMD kernel explicitly (newKernel may or may not select it
// depending on the machine running the suite).
func TestArchKernelDeterministicAcrossThreads(t *testing.T) {
	if newArchKernel(1) == nil {
		t.Skip("no arch kernel on this machine")
	}
	rng := rand.New(rand.NewSource(31))
	m, n, k := 37, 53, 129
	a := randSlice(rng, m*k)
	b16 := f16Bits(randSlice(rng, n*k))
	ref := make([]float32, m*n)
	newArchKernel(1).MatMulTF16(ref, a, b16, m, n, k, nil)
	for _, threads := range []int{2, 4, 16} {
		got := make([]float32, m*n)
		newArchKernel(threads).MatMulTF16(got, a, b16, m, n, k, nil)
		for i := range ref {
			if got[i] != ref[i] {
				t.Fatalf("threads=%d: element %d differs bitwise", threads, i)
			}
		}
	}
}
