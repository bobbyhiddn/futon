package native

import "math"

// float32 transcendentals for the forward pass (softmax, GELU, SiLU).
//
// expF32 is adapted from aikit's linalg/exp.go (github.com/townsendmerino/aikit,
// MIT, Copyright (c) 2026 Francis Townsend-Merino); see NOTICE.
//
// They replace float64 math.Exp/math.Erf round trips, which dominated the
// non-GEMM share of a forward; plain float32 arithmetic runs about twice as
// fast as math.Exp. Like the rest of the forward they are deterministic on a
// given architecture but may differ in the last bit across architectures
// (Go fuses multiply-adds on arm64, not on amd64); aikit's FMA-pinned
// "contract" variants trade that speed back for cross-arch bit identity.
//
// Accuracy (pinned by mathf32_test.go): expF32 is within 1 ULP (it flushes
// to zero below expUnderflowF32, which softmax never observes); geluF32's
// absolute error is ≤ 1e-6, the bound that matters as it feeds a
// k≈1536–3072 GEMM.

const (
	expOverflowF32  = 88.72283  // ln(MaxFloat32)
	expUnderflowF32 = -87.33654 // ln(2^-126): below it, flush to zero
	// expClampLo keeps an exp argument finite (exp of it flushes to 0), so
	// -Inf inputs never reach the float→int conversion.
	expClampLo = -104.0

	log2eF32 = 1.4426950408889634 // 1/ln2
	// ln2 split so k·ln2Hi is exact in f32 for the |k| ≤ 150 seen here.
	ln2HiF32 = 0.693145751953125
	ln2LoF32 = 1.428606820309417e-06
	// 1.5·2^23: adding then subtracting it rounds to the nearest integer.
	roundMagicF32 = 12582912.0
)

// expF32 returns e^x for finite x ≤ expOverflowF32 (callers bound their
// inputs: softmax arguments are ≤ 0, sigmoid's are clamped). Below
// expUnderflowF32 it returns 0.
func expF32(x float32) float32 {
	if x < expUnderflowF32 {
		return 0
	}
	// Range-reduce x = k·ln2 + r, |r| ≤ ln2/2.
	kf := (x*log2eF32 + roundMagicF32) - roundMagicF32
	k := int32(kf)
	r := x - kf*ln2HiF32
	r -= kf * ln2LoF32

	// Cephes single-precision minimax: e^r = 1 + r + r²·P(r).
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 5.0000001201e-1
	p = (p*r+1)*r + 1

	// Scale by 2^k; k = 128 needs two steps to stay a valid exponent.
	e := k + 127
	if e <= 0 {
		return 0
	}
	if e >= 255 {
		return p * math.Float32frombits(uint32(e-1)<<23) * 2
	}
	return p * math.Float32frombits(uint32(e)<<23)
}

// erfcPos is erfc(a) for a ≥ 0: the Chebyshev fit from Numerical Recipes
// (erfcc), fractional error < 1.2e-7 everywhere. Branch-free, unlike a
// split series/tail erf, whose |x| < 1 test mispredicts on activations.
func erfcPos(a float32) float32 {
	t := 1 / (1 + 0.5*a)
	p := float32(0.17087277)
	p = p*t - 0.82215223
	p = p*t + 1.48851587
	p = p*t - 1.13520398
	p = p*t + 0.27886807
	p = p*t - 0.18628806
	p = p*t + 0.09678418
	p = p*t + 0.37409196
	p = p*t + 1.00002368
	p = p*t - 1.26551223
	e := p - a*a
	if e < expClampLo {
		e = expClampLo
	}
	return t * expF32(e)
}

// geluF32 is the exact (erf-based) GELU used by HF's "gelu" activation,
// 0.5·x·(1 + erf(x/√2)), evaluated through erfc: 0.5·x·erfc(|x|/√2) for
// x < 0 (relatively accurate as it vanishes) and 0.5·x·(2 − erfc(x/√2))
// otherwise. Max absolute error 3.7e-7.
func geluF32(x float32) float32 {
	const invSqrt2 = 0.7071067811865476
	a := x * invSqrt2
	if a < 0 {
		a = -a
	}
	c := erfcPos(a)
	if x < 0 {
		return 0.5 * x * c
	}
	return 0.5 * x * (2 - c)
}

// sigmoidF32 is 1/(1+e^−x). The exp argument is clamped to stay finite and
// within expF32's range; at either clamp the result is already saturated.
func sigmoidF32(x float32) float32 {
	e := -x
	if e < expClampLo {
		e = expClampLo
	}
	if e > expOverflowF32 {
		return 0
	}
	return 1 / (1 + expF32(e))
}
