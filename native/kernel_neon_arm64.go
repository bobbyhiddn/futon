//go:build arm64

package native

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// dot4x2f32neon computes the 4×2 dot-product panel out[r*2+c] = Σ a_r[p]·b_c[p]
// for p in [0,k). k must be a positive multiple of 8. Rows may alias
// (the wrapper points remainder rows at a valid row and discards outputs).
//
//go:noescape
func dot4x2f32neon(a0, a1, a2, a3, b0, b1 *float32, k int, out *[8]float32)

// dot4x2f16neon is dot4x2f32neon with b held as IEEE 754 binary16 bits,
// widened in-register (FCVTL — exactly) before the FMA.
//
//go:noescape
func dot4x2f16neon(a0, a1, a2, a3 *float32, b0, b1 *uint16, k int, out *[8]float32)

// newArchKernel returns the NEON kernel when the CPU supports it, else nil
// (the caller falls back to the portable goKernel).
//
// ⚠ HARDWARE VALIDATION STATUS: this kernel was developed on amd64 and
// validated under qemu-aarch64 user-mode emulation (kernel-vs-reference GEMM
// tests + full toy-model architecture suite). It has NOT yet run on real
// arm64 silicon. The kernel correctness tests (TestKernelMatMulTCorrectness,
// TestArchKernelMatchesScalar, TestArchKernelDeterministicAcrossThreads)
// compare against the in-process pure-Go reference and run UNCONDITIONALLY
// on arm64, so the first `go test ./internal/embed/native/...` on real
// hardware validates it automatically — run that, plus the real-weight
// parity suite, before any arm64 production use.
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

func (g *neonKernel) MatMulT(dst, a, b []float32, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	k8 := k &^ 7
	parallelRows(m, g.threads, 4, func(lo, hi int) {
		var out [8]float32
		for i := lo; i < hi; i += 4 {
			last := hi - 1
			a0 := &a[i*k]
			a1 := &a[min(i+1, last)*k]
			a2 := &a[min(i+2, last)*k]
			a3 := &a[min(i+3, last)*k]
			rows := min(4, hi-i)
			for j := 0; j < n; j += 2 {
				j1 := j + 1
				if j1 >= n {
					j1 = j
				}
				if k8 > 0 {
					dot4x2f32neon(a0, a1, a2, a3, &b[j*k], &b[j1*k], k8, &out)
				} else {
					out = [8]float32{}
				}
				bj := b[j*k : j*k+k]
				bj1 := b[j1*k : j1*k+k]
				for r := 0; r < rows; r++ {
					ar := a[(i+r)*k : (i+r)*k+k]
					s0, s1 := out[2*r], out[2*r+1]
					for p := k8; p < k; p++ {
						s0 += ar[p] * bj[p]
						s1 += ar[p] * bj1[p]
					}
					if bias != nil {
						s0 += bias[j]
						s1 += bias[j1]
					}
					dst[(i+r)*n+j] = s0
					if j1 != j {
						dst[(i+r)*n+j1] = s1
					}
				}
			}
		}
	})
}

func (g *neonKernel) MatMulTF16(dst, a []float32, b []uint16, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	k8 := k &^ 7
	parallelRows(m, g.threads, 4, func(lo, hi int) {
		var out [8]float32
		for i := lo; i < hi; i += 4 {
			last := hi - 1
			a0 := &a[i*k]
			a1 := &a[min(i+1, last)*k]
			a2 := &a[min(i+2, last)*k]
			a3 := &a[min(i+3, last)*k]
			rows := min(4, hi-i)
			for j := 0; j < n; j += 2 {
				j1 := j + 1
				if j1 >= n {
					j1 = j
				}
				if k8 > 0 {
					dot4x2f16neon(a0, a1, a2, a3, &b[j*k], &b[j1*k], k8, &out)
				} else {
					out = [8]float32{}
				}
				bj := b[j*k : j*k+k]
				bj1 := b[j1*k : j1*k+k]
				for r := 0; r < rows; r++ {
					ar := a[(i+r)*k : (i+r)*k+k]
					s0, s1 := out[2*r], out[2*r+1]
					for p := k8; p < k; p++ {
						s0 += ar[p] * f16ToF32(bj[p])
						s1 += ar[p] * f16ToF32(bj1[p])
					}
					if bias != nil {
						s0 += bias[j]
						s1 += bias[j1]
					}
					dst[(i+r)*n+j] = s0
					if j1 != j {
						dst[(i+r)*n+j1] = s1
					}
				}
			}
		}
	})
}
