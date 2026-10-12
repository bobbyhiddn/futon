package native

import (
	"os"
	"runtime"
	"sync"
)

// Kernel is the seam every dense-linear-algebra call in the encoder goes
// through. The forward pass never multiplies matrices directly — it always
// calls a Kernel — so per-architecture implementations (the AVX2/FMA/F16C
// kernel in kernel_avx2_amd64.s, future NEON) slot in via newKernel without
// touching any caller.
//
// Layout contract (chosen to be dot-product friendly, which is what SIMD
// kernels want): both operands are row-major and the second operand is
// pre-transposed — exactly PyTorch's nn.Linear weight layout, so weights are
// used as stored.
//
// Determinism contract (pinned by TestKernelDeterministicAcrossThreads):
// for a fixed Kernel implementation on a fixed host, thread count may decide
// WHICH worker computes an output element, never the reduction order within
// one — same input ⇒ bit-identical output at any parallelism. Different
// implementations (scalar vs SIMD) may round differently; cross-kernel and
// cross-host correctness is carried by the cosine parity gates.
type Kernel interface {
	// MatMulT computes dst = a · bᵀ (+ bias broadcast over rows).
	//
	//	a:    m×k row-major float32
	//	b:    n×k row-major float32 (each row is one output feature)
	//	dst:  m×n row-major
	//	bias: nil, or length n
	//
	// dst must not alias a or b.
	MatMulT(dst, a, b []float32, m, n, k int, bias []float32)

	// MatMulTF16 is MatMulT with the second operand held as raw IEEE 754
	// binary16 bits (the pack's on-disk weight format). Conversion to f32
	// happens inside the kernel — in-register on SIMD implementations —
	// and is exact, so results equal MatMulT over the dequantized matrix
	// up to summation order.
	MatMulTF16(dst, a []float32, b []uint16, m, n, k int, bias []float32)
}

// newKernel returns the best available Kernel for this build/CPU using up to
// threads goroutines (threads <= 1 means fully serial).
// FUTON_NATIVE_PUREGO=1 forces the portable scalar kernel (escape hatch and
// benchmark baseline).
func newKernel(threads int) Kernel {
	if threads < 1 {
		threads = runtime.NumCPU()
	}
	if os.Getenv("FUTON_NATIVE_PUREGO") != "1" {
		if k := newArchKernel(threads); k != nil {
			return k
		}
	}
	return &goKernel{threads: threads}
}

// parallelRows partitions rows [0,m) across up to threads workers in
// contiguous chunks aligned to blockRows (so block micro-kernels never
// straddle a chunk edge) and runs fn(lo,hi) on each.
func parallelRows(m, threads, blockRows int, fn func(lo, hi int)) {
	if threads > m {
		threads = m
	}
	if threads <= 1 {
		fn(0, m)
		return
	}
	chunk := (m + threads - 1) / threads
	if r := chunk % blockRows; r != 0 {
		chunk += blockRows - r
	}
	var wg sync.WaitGroup
	for lo := 0; lo < m; lo += chunk {
		hi := lo + chunk
		if hi > m {
			hi = m
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// minParallelWork is the GEMM size (m·n·k multiply-adds) below which
// parallelTiles runs serially: fanning goroutines out over a few tens of
// microseconds of work costs more than it saves. Per-head attention GEMMs
// at typical sequence lengths fall under it.
const minParallelWork = 1 << 21

// parallelTiles partitions the m×n output into blocks aligned to blockRows
// and blockCols and runs fn(i0, i1, j0, j1) on each. When there are enough
// column blocks to balance across threads it splits columns, so each worker
// streams only its slice of b (the weights) — a row split makes every
// worker re-read all of b, which dominates at embedding-length m. Otherwise
// it falls back to parallelRows. (The column split follows aikit's
// encoder/parallel.go, github.com/townsendmerino/aikit, MIT; see NOTICE.)
// Either way each output element is computed whole by one worker, so the
// determinism contract holds.
func parallelTiles(m, n, k, threads, blockRows, blockCols int, fn func(i0, i1, j0, j1 int)) {
	if threads <= 1 || m*n*k < minParallelWork {
		fn(0, m, 0, n)
		return
	}
	colBlocks := (n + blockCols - 1) / blockCols
	if colBlocks < 2*threads {
		parallelRows(m, threads, blockRows, func(lo, hi int) { fn(lo, hi, 0, n) })
		return
	}
	chunk := (colBlocks + threads - 1) / threads * blockCols
	var wg sync.WaitGroup
	for j0 := 0; j0 < n; j0 += chunk {
		j1 := min(j0+chunk, n)
		wg.Add(1)
		go func(j0, j1 int) {
			defer wg.Done()
			fn(0, m, j0, j1)
		}(j0, j1)
	}
	wg.Wait()
}

// threaded is implemented by every built-in kernel. It exposes the thread
// budget, so the forward can also parallelize work outside the GEMMs, and
// makes a same-implementation copy with another budget (head-parallel
// attention runs its small inner GEMMs on serial copies). A Kernel without
// it is treated as serial for everything outside its own GEMMs.
type threaded interface {
	threadCount() int
	withThreads(n int) Kernel
}

// kernelThreads returns kern's thread budget, 1 if it does not say.
func kernelThreads(kern Kernel) int {
	if t, ok := kern.(threaded); ok {
		return t.threadCount()
	}
	return 1
}

// minParallelElems is the elementwise-loop size (activations) below which
// parallelChunks runs serially.
const minParallelElems = 1 << 14

// parallelChunks splits [0,n) into up to threads contiguous chunks of at
// least minParallelElems and runs fn on each.
func parallelChunks(n, threads int, fn func(lo, hi int)) {
	threads = min(threads, n/minParallelElems)
	if threads <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + threads - 1) / threads
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

func checkGemmArgs(lenDst, lenA, lenB int, m, n, k int, bias []float32) {
	if lenA < m*k || lenB < n*k || lenDst < m*n {
		panic("native: MatMulT buffer smaller than declared shape")
	}
	if bias != nil && len(bias) < n {
		panic("native: MatMulT bias smaller than n")
	}
}

// goKernel is the portable blocked, goroutine-parallel pure-Go f32 GEMM.
//
// The micro-kernel computes a 4-row × 1-col block per inner loop so each row
// of b (one output feature, k contiguous floats) is streamed once per four
// rows of a instead of once per row — a 4× cut in memory traffic on the
// dominant operand. The inner dot products are 4-way unrolled.
type goKernel struct {
	threads int
}

func (g *goKernel) threadCount() int         { return g.threads }
func (g *goKernel) withThreads(n int) Kernel { return &goKernel{threads: n} }

func (g *goKernel) MatMulT(dst, a, b []float32, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	parallelRows(m, g.threads, 4, func(lo, hi int) {
		matMulTRows(dst, a, b, lo, hi, n, k, bias)
	})
}

func (g *goKernel) MatMulTF16(dst, a []float32, b []uint16, m, n, k int, bias []float32) {
	checkGemmArgs(len(dst), len(a), len(b), m, n, k, bias)
	parallelRows(m, g.threads, 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ai := a[i*k : i*k+k]
			di := dst[i*n : i*n+n]
			for j := 0; j < n; j++ {
				bj := b[j*k : j*k+k]
				var s float32
				for p := 0; p < k; p++ {
					s += ai[p] * f16ToF32(bj[p])
				}
				if bias != nil {
					s += bias[j]
				}
				di[j] = s
			}
		}
	})
}

// matMulTRows computes rows [lo,hi) of dst = a·bᵀ + bias.
func matMulTRows(dst, a, b []float32, lo, hi, n, k int, bias []float32) {
	i := lo
	// 4-row micro-kernel.
	for ; i+4 <= hi; i += 4 {
		a0 := a[i*k : i*k+k]
		a1 := a[(i+1)*k : (i+1)*k+k]
		a2 := a[(i+2)*k : (i+2)*k+k]
		a3 := a[(i+3)*k : (i+3)*k+k]
		d0 := dst[i*n : i*n+n]
		d1 := dst[(i+1)*n : (i+1)*n+n]
		d2 := dst[(i+2)*n : (i+2)*n+n]
		d3 := dst[(i+3)*n : (i+3)*n+n]
		for j := 0; j < n; j++ {
			bj := b[j*k : j*k+k]
			var s0, s1, s2, s3 float32
			p := 0
			for ; p+4 <= k; p += 4 {
				bv0, bv1, bv2, bv3 := bj[p], bj[p+1], bj[p+2], bj[p+3]
				s0 += a0[p]*bv0 + a0[p+1]*bv1 + a0[p+2]*bv2 + a0[p+3]*bv3
				s1 += a1[p]*bv0 + a1[p+1]*bv1 + a1[p+2]*bv2 + a1[p+3]*bv3
				s2 += a2[p]*bv0 + a2[p+1]*bv1 + a2[p+2]*bv2 + a2[p+3]*bv3
				s3 += a3[p]*bv0 + a3[p+1]*bv1 + a3[p+2]*bv2 + a3[p+3]*bv3
			}
			for ; p < k; p++ {
				bv := bj[p]
				s0 += a0[p] * bv
				s1 += a1[p] * bv
				s2 += a2[p] * bv
				s3 += a3[p] * bv
			}
			if bias != nil {
				bb := bias[j]
				s0 += bb
				s1 += bb
				s2 += bb
				s3 += bb
			}
			d0[j], d1[j], d2[j], d3[j] = s0, s1, s2, s3
		}
	}
	// Remainder rows.
	for ; i < hi; i++ {
		ai := a[i*k : i*k+k]
		di := dst[i*n : i*n+n]
		for j := 0; j < n; j++ {
			bj := b[j*k : j*k+k]
			var s float32
			p := 0
			for ; p+4 <= k; p += 4 {
				s += ai[p]*bj[p] + ai[p+1]*bj[p+1] + ai[p+2]*bj[p+2] + ai[p+3]*bj[p+3]
			}
			for ; p < k; p++ {
				s += ai[p] * bj[p]
			}
			if bias != nil {
				s += bias[j]
			}
			di[j] = s
		}
	}
}
