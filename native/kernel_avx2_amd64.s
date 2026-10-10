// AVX2/FMA/F16C micro-kernels for the native embedding GEMM.
//
// Both routines compute an 8-element panel of dot products: four rows of A
// against two rows of B (B is pre-transposed, so "row of B" = one output
// feature), accumulating over k in 8-float lanes with one FMA chain per
// (row, col) pair — 8 independent chains, enough to hide FMA latency on
// small cores. The horizontal reduction sequence is fixed, so results are
// bit-deterministic for a given k regardless of caller threading (the
// determinism contract in kernel.go).
//
// The f16 variant reads B as IEEE 754 binary16 and converts in-register
// (VCVTPH2PS) — exact conversion, half the memory traffic of f32 weights.
//
// k must be a positive multiple of 8; the Go wrapper handles k<8 and tails.

#include "textflag.h"

// REDUCE1 folds one 8-lane accumulator YR (low half XR) to a scalar and
// stores it at OFF(DI). Clobbers X8. Fixed sequence — determinism contract.
#define REDUCE1(YR, XR, OFF) \
	VEXTRACTF128 $1, YR, X8; \
	VADDPS       X8, XR, XR; \
	VHADDPS      XR, XR, XR; \
	VHADDPS      XR, XR, XR; \
	VMOVSS       XR, OFF(DI)

#define ZERO_ACC \
	VXORPS Y0, Y0, Y0; \
	VXORPS Y1, Y1, Y1; \
	VXORPS Y2, Y2, Y2; \
	VXORPS Y3, Y3, Y3; \
	VXORPS Y4, Y4, Y4; \
	VXORPS Y5, Y5, Y5; \
	VXORPS Y6, Y6, Y6; \
	VXORPS Y7, Y7, Y7

#define LOAD_ARGS \
	MOVQ a0+0(FP), R8; \
	MOVQ a1+8(FP), R9; \
	MOVQ a2+16(FP), R10; \
	MOVQ a3+24(FP), R11; \
	MOVQ b0+32(FP), R12; \
	MOVQ b1+40(FP), R13; \
	MOVQ k+48(FP), CX; \
	MOVQ out+56(FP), DI

// FMA_BLOCK multiplies the two loaded B vectors (Y8, Y9) against all four A
// rows and advances the A pointers by 8 floats.
#define FMA_BLOCK \
	VMOVUPS     (R8), Y10; \
	VFMADD231PS Y8, Y10, Y0; \
	VFMADD231PS Y9, Y10, Y1; \
	VMOVUPS     (R9), Y11; \
	VFMADD231PS Y8, Y11, Y2; \
	VFMADD231PS Y9, Y11, Y3; \
	VMOVUPS     (R10), Y12; \
	VFMADD231PS Y8, Y12, Y4; \
	VFMADD231PS Y9, Y12, Y5; \
	VMOVUPS     (R11), Y13; \
	VFMADD231PS Y8, Y13, Y6; \
	VFMADD231PS Y9, Y13, Y7; \
	ADDQ        $32, R8; \
	ADDQ        $32, R9; \
	ADDQ        $32, R10; \
	ADDQ        $32, R11

#define REDUCE_ALL \
	REDUCE1(Y0, X0, 0); \
	REDUCE1(Y1, X1, 4); \
	REDUCE1(Y2, X2, 8); \
	REDUCE1(Y3, X3, 12); \
	REDUCE1(Y4, X4, 16); \
	REDUCE1(Y5, X5, 20); \
	REDUCE1(Y6, X6, 24); \
	REDUCE1(Y7, X7, 28)

// func dot4x2f32(a0, a1, a2, a3, b0, b1 *float32, k int, out *[8]float32)
TEXT ·dot4x2f32(SB), NOSPLIT, $0-64
	LOAD_ARGS
	ZERO_ACC

f32loop:
	VMOVUPS (R12), Y8  // b0[p:p+8]
	VMOVUPS (R13), Y9  // b1[p:p+8]
	FMA_BLOCK
	ADDQ    $32, R12
	ADDQ    $32, R13
	SUBQ    $8, CX
	JNZ     f32loop

	REDUCE_ALL
	VZEROUPPER
	RET

// func dot4x2f16(a0, a1, a2, a3 *float32, b0, b1 *uint16, k int, out *[8]float32)
TEXT ·dot4x2f16(SB), NOSPLIT, $0-64
	LOAD_ARGS
	ZERO_ACC

f16loop:
	VCVTPH2PS (R12), Y8  // b0[p:p+8] f16→f32 (exact)
	VCVTPH2PS (R13), Y9  // b1[p:p+8]
	FMA_BLOCK
	ADDQ      $16, R12
	ADDQ      $16, R13
	SUBQ      $8, CX
	JNZ       f16loop

	REDUCE_ALL
	VZEROUPPER
	RET

// func cpuidLeaf1ECX() uint32
TEXT ·cpuidLeaf1ECX(SB), NOSPLIT, $0-4
	MOVL $1, AX
	XORL CX, CX
	CPUID
	MOVL CX, ret+0(FP)
	RET
