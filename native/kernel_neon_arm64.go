//go:build arm64

package native

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// dot4x4f32neon computes the 4×4 dot-product panel out[r*4+c] = Σ a_r[p]·b_c[p]
// for p in [0,k). k must be a positive multiple of 8. Rows and columns may
// alias (the wrapper points edge rows/columns at a valid one and discards
// their outputs).
//
//go:noescape
func dot4x4f32neon(a0, a1, a2, a3, b0, b1, b2, b3 *float32, k int, out *[16]float32)

// dot4x4f16neon is dot4x4f32neon with b held as IEEE 754 binary16 bits,
// widened in-register (FCVTL — exactly) before the FMA.
//
//go:noescape
func dot4x4f16neon(a0, a1, a2, a3 *float32, b0, b1, b2, b3 *uint16, k int, out *[16]float32)

// newArchKernel returns the NEON kernel when the CPU supports it, else nil
// (the caller falls back to the portable goKernel).
//
// Validated on Apple silicon (M4 Pro): the kernel correctness and
// determinism tests, which compare against the in-process pure-Go reference
// and run unconditionally on arm64, plus real-weight parity on every pinned
// pack.
func newArchKernel(threads int) Kernel {
	if !neonAvailable() {
		return nil
	}
	return &neonKernel{threads: threads}
}

// neonAvailable reports whether the NEON (ASIMD) micro-kernels can run.
// ASIMD and binary16→binary32 widening (FCVTL) are baseline ARMv8-A — no
// optional extension (SVE, FEAT_FP16 arithmetic) is used — so this is
// effectively always true; the check is a guard against exotic
// float-less/ASIMD-less profiles, with the pure-Go kernel as fallback.
// darwin/arm64 (Apple silicon) guarantees ASIMD architecturally and does not
// always populate the HWCAP-derived flags, so it is trusted directly.
func neonAvailable() bool {
	return runtime.GOOS == "darwin" || cpu.ARM64.HasASIMD
}

// neonKernel implements Kernel with the assembly micro-kernels in
// kernel_neon_arm64.s. Same determinism contract as goKernel: thread count
// never changes any output bit (fixed lane accumulation + fixed reduction
// sequence per element; scalar k-tails are added after the SIMD sum in a
// fixed order).
type neonKernel struct {
	threads int
}

func (g *neonKernel) threadCount() int         { return g.threads }
func (g *neonKernel) withThreads(n int) Kernel { return &neonKernel{threads: n} }

func (g *neonKernel) MatMulT(dst, a, b []float32, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	neonMatMul(dst, a, b, m, n, k, bias, g.threads, dot4x4f32neon, func(v float32) float32 { return v })
}

func (g *neonKernel) MatMulTF16(dst, a []float32, b []uint16, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	neonMatMul(dst, a, b, m, n, k, bias, g.threads, dot4x4f16neon, f16ToF32)
}

// neonMatMul tiles dst = a·bᵀ + bias into 4×4 panels for the assembly
// kernel, split across threads by parallelTiles. Edge panels alias their last valid row/column and drop the
// duplicate outputs; the k%8 tail is added per element after the SIMD sum,
// in a fixed order.
func neonMatMul[T float32 | uint16](dst, a []float32, b []T, m, n, k int, bias []float32, threads int,
	panel func(a0, a1, a2, a3 *float32, b0, b1, b2, b3 *T, k int, out *[16]float32), conv func(T) float32) {
	k8 := k &^ 7
	parallelTiles(m, n, k, threads, 4, 4, func(i0, i1, j0, j1 int) {
		var out [16]float32
		for i := i0; i < i1; i += 4 {
			rows := min(4, i1-i)
			last := i1 - 1
			a0 := &a[i*k]
			a1 := &a[min(i+1, last)*k]
			a2 := &a[min(i+2, last)*k]
			a3 := &a[min(i+3, last)*k]
			for j := j0; j < j1; j += 4 {
				cols := min(4, j1-j)
				var jc [4]int
				for c := range jc {
					jc[c] = min(j+c, j1-1)
				}
				if k8 > 0 {
					panel(a0, a1, a2, a3, &b[jc[0]*k], &b[jc[1]*k], &b[jc[2]*k], &b[jc[3]*k], k8, &out)
				} else {
					out = [16]float32{}
				}
				if rows == 4 && cols == 4 && k8 == k {
					// Full panel, no k tail: the common case, kept free of
					// per-element loops (small-k attention GEMMs are
					// epilogue-bound otherwise).
					for r := 0; r < 4; r++ {
						dr := dst[(i+r)*n+j : (i+r)*n+j+4]
						or := out[r*4 : r*4+4]
						if bias != nil {
							br := bias[j : j+4]
							dr[0], dr[1], dr[2], dr[3] = or[0]+br[0], or[1]+br[1], or[2]+br[2], or[3]+br[3]
						} else {
							copy(dr, or)
						}
					}
					continue
				}
				for r := 0; r < rows; r++ {
					ar := a[(i+r)*k : (i+r)*k+k]
					dr := dst[(i+r)*n+j : (i+r)*n+j+cols]
					for c := 0; c < cols; c++ {
						s := out[r*4+c]
						bc := b[jc[c]*k : jc[c]*k+k]
						for p := k8; p < k; p++ {
							s += ar[p] * conv(bc[p])
						}
						if bias != nil {
							s += bias[j+c]
						}
						dr[c] = s
					}
				}
			}
		}
	})
}
