//go:build arm64

package native

// MatMulTQ8 gives the NEON kernel the experimental Q8 entry point (see
// quant.go). The integer GEMM itself is the shared portable implementation.
func (g *neonKernel) MatMulTQ8(dst, a []float32, qb []int8, scales []float32, m, n, k int, bias []float32) {
	matMulTQ8(dst, a, qb, scales, m, n, k, bias, g.threads)
}
