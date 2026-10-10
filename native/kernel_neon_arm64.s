// NEON micro-kernels for the native embedding GEMM (arm64).
//
// Mirrors kernel_avx2_amd64.s: both routines compute an 8-element panel of
// dot products — four rows of A against two rows of B (B pre-transposed, so
// "row of B" = one output feature) — accumulating over k in 4-float NEON
// lanes. Each (row, col) pair owns ONE accumulator register; every loop
// iteration folds 8 k-elements into it as two FMLA steps in a fixed order
// (low 4 floats, then high 4), and the horizontal reduction sequence is
// fixed (FADDP pairwise, then scalar-pair FADDP), so results are
// bit-deterministic for a given k regardless of caller threading (the
// determinism contract in kernel.go).
//
// The f16 variant reads B as IEEE 754 binary16 and widens in-register
// (FCVTL/FCVTL2 — exact conversion, half the memory traffic of f32 weights).
// binary16→binary32 widening is baseline ARMv8-A FP; no optional extension
// (FEAT_FP16 arithmetic) is used anywhere.
//
// k must be a positive multiple of 8; the Go wrapper handles k<8 and tails.
//
// Go's assembler has no mnemonics for FADDP or FCVTL, so those four
// instructions are WORD-encoded with fixed registers (encodings from the
// Arm ARM; each use site says which instruction it is).

#include "textflag.h"

// REDUCE1 folds accumulator V<r> to a scalar in S<r> (= F<r>, passed as FR
// because the assembler has no token pasting) and stores it at OFF(R15).
// Fixed sequence — determinism contract. The two WORDs are, in order:
//   FADDP V<r>.4S, V<r>.4S, V<r>.4S   (0x6E20D400 | r<<16 | r<<5 | r)
//   FADDP S<r>, V<r>.2S               (0x7E30D800 | r<<5  | r)
#define REDUCE1(r, FR, OFF) \
	WORD  $(0x6E20D400 | r<<16 | r<<5 | r); \
	WORD  $(0x7E30D800 | r<<5 | r); \
	FMOVS FR, OFF(R15)

#define ZERO_ACC \
	VEOR V0.B16, V0.B16, V0.B16; \
	VEOR V1.B16, V1.B16, V1.B16; \
	VEOR V2.B16, V2.B16, V2.B16; \
	VEOR V3.B16, V3.B16, V3.B16; \
	VEOR V4.B16, V4.B16, V4.B16; \
	VEOR V5.B16, V5.B16, V5.B16; \
	VEOR V6.B16, V6.B16, V6.B16; \
	VEOR V7.B16, V7.B16, V7.B16

#define LOAD_ARGS \
	MOVD a0+0(FP), R8; \
	MOVD a1+8(FP), R9; \
	MOVD a2+16(FP), R10; \
	MOVD a3+24(FP), R11; \
	MOVD b0+32(FP), R12; \
	MOVD b1+40(FP), R13; \
	MOVD k+48(FP), R14; \
	MOVD out+56(FP), R15

// FMA_BLOCK multiplies the widened B vectors (b0 in V8:V9, b1 in V10:V11 —
// low 4 k-elements then high 4) against all four A rows, advancing the A
// pointers by 8 floats. Per accumulator the order is fixed: low block, then
// high block.
#define FMA_BLOCK \
	VLD1.P 32(R8), [V12.S4, V13.S4]; \
	VFMLA  V8.S4, V12.S4, V0.S4; \
	VFMLA  V10.S4, V12.S4, V1.S4; \
	VFMLA  V9.S4, V13.S4, V0.S4; \
	VFMLA  V11.S4, V13.S4, V1.S4; \
	VLD1.P 32(R9), [V14.S4, V15.S4]; \
	VFMLA  V8.S4, V14.S4, V2.S4; \
	VFMLA  V10.S4, V14.S4, V3.S4; \
	VFMLA  V9.S4, V15.S4, V2.S4; \
	VFMLA  V11.S4, V15.S4, V3.S4; \
	VLD1.P 32(R10), [V16.S4, V17.S4]; \
	VFMLA  V8.S4, V16.S4, V4.S4; \
	VFMLA  V10.S4, V16.S4, V5.S4; \
	VFMLA  V9.S4, V17.S4, V4.S4; \
	VFMLA  V11.S4, V17.S4, V5.S4; \
	VLD1.P 32(R11), [V18.S4, V19.S4]; \
	VFMLA  V8.S4, V18.S4, V6.S4; \
	VFMLA  V10.S4, V18.S4, V7.S4; \
	VFMLA  V9.S4, V19.S4, V6.S4; \
	VFMLA  V11.S4, V19.S4, V7.S4

#define REDUCE_ALL \
	REDUCE1(0, F0, 0); \
	REDUCE1(1, F1, 4); \
	REDUCE1(2, F2, 8); \
	REDUCE1(3, F3, 12); \
	REDUCE1(4, F4, 16); \
	REDUCE1(5, F5, 20); \
	REDUCE1(6, F6, 24); \
	REDUCE1(7, F7, 28)

// func dot4x2f32neon(a0, a1, a2, a3, b0, b1 *float32, k int, out *[8]float32)
TEXT ·dot4x2f32neon(SB), NOSPLIT, $0-64
	LOAD_ARGS
	ZERO_ACC

f32loop:
	VLD1.P 32(R12), [V8.S4, V9.S4]   // b0[p:p+8]
	VLD1.P 32(R13), [V10.S4, V11.S4] // b1[p:p+8]
	FMA_BLOCK
	SUBS   $8, R14
	BNE    f32loop

	REDUCE_ALL
	RET

// FCVTL  Vd.4S, Vn.4H = 0x0E217800 | n<<5 | d   (widen low 4 halves)
// FCVTL2 Vd.4S, Vn.8H = 0x4E217800 | n<<5 | d   (widen high 4 halves)
//
// func dot4x2f16neon(a0, a1, a2, a3 *float32, b0, b1 *uint16, k int, out *[8]float32)
TEXT ·dot4x2f16neon(SB), NOSPLIT, $0-64
	LOAD_ARGS
	ZERO_ACC

f16loop:
	VLD1.P 16(R12), [V20.H8]           // b0[p:p+8] as binary16
	WORD   $(0x0E217800 | 20<<5 | 8)   // FCVTL  V8.4S, V20.4H (exact)
	WORD   $(0x4E217800 | 20<<5 | 9)   // FCVTL2 V9.4S, V20.8H
	VLD1.P 16(R13), [V21.H8]           // b1[p:p+8]
	WORD   $(0x0E217800 | 21<<5 | 10)  // FCVTL  V10.4S, V21.4H
	WORD   $(0x4E217800 | 21<<5 | 11)  // FCVTL2 V11.4S, V21.8H
	FMA_BLOCK
	SUBS   $8, R14
	BNE    f16loop

	REDUCE_ALL
	RET
