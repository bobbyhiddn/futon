// NEON micro-kernels for the native embedding GEMM (arm64).
//
// Both routines compute a 16-element panel of dot products — four rows of A
// against four rows of B (B pre-transposed, so "row of B" = one output
// feature) — accumulating over k in 4-float NEON lanes. Sixteen independent
// accumulators keep all four FMA pipes of a wide core (Apple M-series) busy;
// the earlier 4×2 panel's eight chains were FMA-latency bound. (The panel
// shape follows aikit's measurement that ≥ 10 independent chains are needed
// to reach FMA peak — github.com/townsendmerino/aikit, MIT; see NOTICE.)
//
// Each (row, col) pair owns ONE accumulator register; every loop iteration
// folds 8 k-elements into it as two FMLA steps in a fixed order (low 4
// floats, then high 4), and the horizontal reduction sequence is fixed
// (FADDP pairwise, then scalar-pair FADDP), so results are bit-deterministic
// for a given k regardless of caller threading (the determinism contract in
// kernel.go).
//
// Register map: V0–V15 accumulators (V[r*4+c]); V16–V23 the widened B
// columns (column c low half in V[16+2c], high half in V[17+2c]); V24–V27
// two alternating A-row buffers.
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
// because the assembler has no token pasting) and stores it at OFF(R20).
// Fixed sequence — determinism contract. The two WORDs are, in order:
//   FADDP V<r>.4S, V<r>.4S, V<r>.4S   (0x6E20D400 | r<<16 | r<<5 | r)
//   FADDP S<r>, V<r>.2S               (0x7E30D800 | r<<5  | r)
#define REDUCE1(r, FR, OFF) \
	WORD  $(0x6E20D400 | r<<16 | r<<5 | r); \
	WORD  $(0x7E30D800 | r<<5 | r); \
	FMOVS FR, OFF(R20)

#define ZERO_ACC \
	VEOR V0.B16, V0.B16, V0.B16; \
	VEOR V1.B16, V1.B16, V1.B16; \
	VEOR V2.B16, V2.B16, V2.B16; \
	VEOR V3.B16, V3.B16, V3.B16; \
	VEOR V4.B16, V4.B16, V4.B16; \
	VEOR V5.B16, V5.B16, V5.B16; \
	VEOR V6.B16, V6.B16, V6.B16; \
	VEOR V7.B16, V7.B16, V7.B16; \
	VEOR V8.B16, V8.B16, V8.B16; \
	VEOR V9.B16, V9.B16, V9.B16; \
	VEOR V10.B16, V10.B16, V10.B16; \
	VEOR V11.B16, V11.B16, V11.B16; \
	VEOR V12.B16, V12.B16, V12.B16; \
	VEOR V13.B16, V13.B16, V13.B16; \
	VEOR V14.B16, V14.B16, V14.B16; \
	VEOR V15.B16, V15.B16, V15.B16

#define LOAD_ARGS \
	MOVD a0+0(FP), R8; \
	MOVD a1+8(FP), R9; \
	MOVD a2+16(FP), R10; \
	MOVD a3+24(FP), R11; \
	MOVD b0+32(FP), R12; \
	MOVD b1+40(FP), R13; \
	MOVD b2+48(FP), R14; \
	MOVD b3+56(FP), R15; \
	MOVD k+64(FP), R19; \
	MOVD out+72(FP), R20

// ROW multiplies one A row (8 floats loaded from RA into VLO:VHI) against
// the four widened B columns, into accumulators V<base..base+3>. Per
// accumulator the order is fixed: low block, then high block.
#define ROW(RA, VLO, VHI, A0, A1, A2, A3) \
	VLD1.P 32(RA), [VLO.S4, VHI.S4]; \
	VFMLA  V16.S4, VLO.S4, A0.S4; \
	VFMLA  V18.S4, VLO.S4, A1.S4; \
	VFMLA  V20.S4, VLO.S4, A2.S4; \
	VFMLA  V22.S4, VLO.S4, A3.S4; \
	VFMLA  V17.S4, VHI.S4, A0.S4; \
	VFMLA  V19.S4, VHI.S4, A1.S4; \
	VFMLA  V21.S4, VHI.S4, A2.S4; \
	VFMLA  V23.S4, VHI.S4, A3.S4

#define FMA_PANEL \
	ROW(R8, V24, V25, V0, V1, V2, V3); \
	ROW(R9, V26, V27, V4, V5, V6, V7); \
	ROW(R10, V24, V25, V8, V9, V10, V11); \
	ROW(R11, V26, V27, V12, V13, V14, V15)

#define REDUCE_ALL \
	REDUCE1(0, F0, 0); \
	REDUCE1(1, F1, 4); \
	REDUCE1(2, F2, 8); \
	REDUCE1(3, F3, 12); \
	REDUCE1(4, F4, 16); \
	REDUCE1(5, F5, 20); \
	REDUCE1(6, F6, 24); \
	REDUCE1(7, F7, 28); \
	REDUCE1(8, F8, 32); \
	REDUCE1(9, F9, 36); \
	REDUCE1(10, F10, 40); \
	REDUCE1(11, F11, 44); \
	REDUCE1(12, F12, 48); \
	REDUCE1(13, F13, 52); \
	REDUCE1(14, F14, 56); \
	REDUCE1(15, F15, 60)

// func dot4x4f32neon(a0, a1, a2, a3, b0, b1, b2, b3 *float32, k int, out *[16]float32)
TEXT ·dot4x4f32neon(SB), NOSPLIT, $0-80
	LOAD_ARGS
	ZERO_ACC

f32loop:
	VLD1.P 32(R12), [V16.S4, V17.S4] // b0[p:p+8]
	VLD1.P 32(R13), [V18.S4, V19.S4] // b1[p:p+8]
	VLD1.P 32(R14), [V20.S4, V21.S4] // b2[p:p+8]
	VLD1.P 32(R15), [V22.S4, V23.S4] // b3[p:p+8]
	FMA_PANEL
	SUBS   $8, R19
	BNE    f32loop

	REDUCE_ALL
	RET

// WIDEN loads 8 binary16 values from RB into V<hi> and widens them exactly:
//   FCVTL  Vd.4S, Vn.4H = 0x0E217800 | n<<5 | d   (low 4 halves → V<lo>)
//   FCVTL2 Vd.4S, Vn.8H = 0x4E217800 | n<<5 | d   (high 4 halves, in place)
#define WIDEN(RB, VHI, lo, hi) \
	VLD1.P 16(RB), [VHI.H8]; \
	WORD   $(0x0E217800 | hi<<5 | lo); \
	WORD   $(0x4E217800 | hi<<5 | hi)

// func dot4x4f16neon(a0, a1, a2, a3 *float32, b0, b1, b2, b3 *uint16, k int, out *[16]float32)
TEXT ·dot4x4f16neon(SB), NOSPLIT, $0-80
	LOAD_ARGS
	ZERO_ACC

f16loop:
	WIDEN(R12, V17, 16, 17) // b0[p:p+8]
	WIDEN(R13, V19, 18, 19) // b1[p:p+8]
	WIDEN(R14, V21, 20, 21) // b2[p:p+8]
	WIDEN(R15, V23, 22, 23) // b3[p:p+8]
	FMA_PANEL
	SUBS   $8, R19
	BNE    f16loop

	REDUCE_ALL
	RET
