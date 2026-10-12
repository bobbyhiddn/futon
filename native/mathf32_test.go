package native

import (
	"math"
	"testing"
)

// TestExpF32Accuracy bounds expF32 against float64 math.Exp over the range
// softmax and sigmoid feed it, excluding the flush-to-zero region.
func TestExpF32Accuracy(t *testing.T) {
	var maxRel float64
	for x := float32(expUnderflowF32 + 0.01); x < expOverflowF32-0.01; x += 0.000731 {
		ref := math.Exp(float64(x))
		rel := math.Abs(float64(expF32(x))-ref) / ref
		maxRel = math.Max(maxRel, rel)
	}
	// One f32 ULP is 2^-23 relative at most.
	if maxRel > 0x1p-23 {
		t.Fatalf("expF32 max relative error %.3g exceeds 1 ULP", maxRel)
	}
	if expF32(0) != 1 {
		t.Fatalf("expF32(0) = %v, want exactly 1", expF32(0))
	}
	if expF32(expClampLo) != 0 {
		t.Fatalf("expF32(%v) = %v, want flush to 0", float32(expClampLo), expF32(expClampLo))
	}
}

// TestGELUF32Accuracy bounds geluF32 against the float64 erf GELU.
func TestGELUF32Accuracy(t *testing.T) {
	var maxAbs float64
	for x := float32(-12); x < 12; x += 0.0000917 {
		ref := 0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2))
		maxAbs = math.Max(maxAbs, math.Abs(float64(geluF32(x))-ref))
	}
	if maxAbs > 1e-6 {
		t.Fatalf("geluF32 max absolute error %.3g exceeds 1e-6", maxAbs)
	}
	for _, x := range []float32{-200, 200} {
		ref := float32(0.5 * float64(x) * (1 + math.Erf(float64(x)/math.Sqrt2)))
		if got := geluF32(x); got != ref {
			t.Fatalf("geluF32(%v) = %v, want %v", x, got, ref)
		}
	}
}

// TestSigmoidF32Accuracy bounds sigmoidF32 (SiLU's gate) including the
// saturated tails.
func TestSigmoidF32Accuracy(t *testing.T) {
	var maxAbs float64
	for x := float32(-120); x < 120; x += 0.00113 {
		ref := 1 / (1 + math.Exp(-float64(x)))
		maxAbs = math.Max(maxAbs, math.Abs(float64(sigmoidF32(x))-ref))
	}
	if maxAbs > 2e-7 {
		t.Fatalf("sigmoidF32 max absolute error %.3g exceeds 2e-7", maxAbs)
	}
}
