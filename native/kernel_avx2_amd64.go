//go:build amd64

package native

import "golang.org/x/sys/cpu"

// dot4x2f32 computes the 4×2 dot-product panel out[r*2+c] = Σ a_r[p]·b_c[p]
// for p in [0,k). k must be a positive multiple of 8. Rows may alias
// (the wrapper points remainder rows at a valid row and discards outputs).
//
//go:noescape
func dot4x2f32(a0, a1, a2, a3, b0, b1 *float32, k int, out *[8]float32)

// dot4x2f16 is dot4x2f32 with b held as IEEE 754 binary16 bits, converted
// in-register (exactly) before the FMA.
//
//go:noescape
func dot4x2f16(a0, a1, a2, a3 *float32, b0, b1 *uint16, k int, out *[8]float32)

// cpuidLeaf1ECX returns ECX of CPUID leaf 1. x/sys/cpu does not surface
// F16C (leaf 1 ECX bit 29), so we read it ourselves.
func cpuidLeaf1ECX() uint32

const cpuidF16CBit = 1 << 29

// newArchKernel returns the AVX2/FMA/F16C kernel when the CPU supports it,
// else nil (the caller falls back to the portable goKernel).
func newArchKernel(threads int) Kernel {
	if cpu.X86.HasAVX2 && cpu.X86.HasFMA && cpuidLeaf1ECX()&cpuidF16CBit != 0 {
		return &avx2Kernel{threads: threads}
	}
	return nil
}

// avx2Kernel implements Kernel with the assembly micro-kernels in
// kernel_avx2_amd64.s, split across threads by parallelTiles. Same determinism contract as goKernel: thread count
// never changes any output bit (fixed lane accumulation + fixed reduction
// sequence per element; scalar k-tails are added after the SIMD sum in a
// fixed order).
type avx2Kernel struct {
	threads int
}

func (g *avx2Kernel) threadCount() int         { return g.threads }
func (g *avx2Kernel) withThreads(n int) Kernel { return &avx2Kernel{threads: n} }

func (g *avx2Kernel) MatMulT(dst, a, b []float32, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	k8 := k &^ 7
	parallelTiles(m, n, k, g.threads, 4, 2, func(lo, hi, c0, c1 int) {
		var out [8]float32
		for i := lo; i < hi; i += 4 {
			last := hi - 1
			a0 := &a[i*k]
			a1 := &a[min(i+1, last)*k]
			a2 := &a[min(i+2, last)*k]
			a3 := &a[min(i+3, last)*k]
			rows := min(4, hi-i)
			for j := c0; j < c1; j += 2 {
				j1 := j + 1
				if j1 >= c1 {
					j1 = j
				}
				if k8 > 0 {
					dot4x2f32(a0, a1, a2, a3, &b[j*k], &b[j1*k], k8, &out)
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

func (g *avx2Kernel) MatMulTF16(dst, a []float32, b []uint16, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	k8 := k &^ 7
	parallelTiles(m, n, k, g.threads, 4, 2, func(lo, hi, c0, c1 int) {
		var out [8]float32
		for i := lo; i < hi; i += 4 {
			last := hi - 1
			a0 := &a[i*k]
			a1 := &a[min(i+1, last)*k]
			a2 := &a[min(i+2, last)*k]
			a3 := &a[min(i+3, last)*k]
			rows := min(4, hi-i)
			for j := c0; j < c1; j += 2 {
				j1 := j + 1
				if j1 >= c1 {
					j1 = j
				}
				if k8 > 0 {
					dot4x2f16(a0, a1, a2, a3, &b[j*k], &b[j1*k], k8, &out)
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
