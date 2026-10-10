package native

import "math"

// Experimental int8 (Q8) weight path — OFF BY DEFAULT, evidence-gathering
// only (Phase 5). Enabled solely by FUTON_NATIVE_Q8=1 at model load; no
// config field, no factory wiring, no default path reaches it.
//
// Scheme: symmetric per-row absmax quantization.
//   - Weights: each row (one output feature, k contiguous values) is
//     quantized once at load to int8 with scale = absmax/127.
//   - Activations: each row of the GEMM input is quantized the same way,
//     dynamically, inside MatMulTQ8.
//   - Dot products accumulate int8×int8 into int32 EXACTLY (k·127² ≤ 2³¹
//     for every k this model family has), then one float32 rescale by
//     scaleA·scaleB per output element.
//
// Determinism: quantization is per-row and elementwise, the integer
// accumulation is exact, and the final rescale is one fixed expression —
// thread count decides which worker computes an element, never the value,
// so the kernel determinism contract holds on this path too (pinned by
// TestQ8DeterministicAcrossThreads).
//
// DECISION RULE (pre-agreed): Q8 ships as an option only if golden parity
// min ≥ 0.999 over the full corpus; otherwise it stays experimental-off
// with the measured numbers recorded in docs/design/native-embeddings.md.

// envNativeQ8 gates the experimental Q8 weight path. See package comment
// above; read once at LoadModel.
const envNativeQ8 = "FUTON_NATIVE_Q8"

// q8Mat is a per-row absmax-quantized matrix: data[j*k+p] holds row j's
// int8 values, scale[j] dequantizes them (value ≈ data·scale).
type q8Mat struct {
	data  []int8
	scale []float32
}

// quantizeRowQ8 quantizes one row into dst (same length) and returns the
// dequantization scale. Symmetric: values map to [-127, 127]; -128 is never
// produced. An all-zero row gets scale 0 and all-zero codes.
func quantizeRowQ8(dst []int8, src []float32) float32 {
	var amax float32
	for _, v := range src {
		if v < 0 {
			v = -v
		}
		if v > amax {
			amax = v
		}
	}
	if amax == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return 0
	}
	scale := amax / 127
	inv := float64(127) / float64(amax)
	for i, v := range src {
		q := math.Round(float64(v) * inv)
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q)
	}
	return scale
}

// quantizeQ8 quantizes an n×k row-major weight matrix (fp16 or f32 storage)
// to per-row int8. Rows are quantized from the exactly-dequantized values,
// so fp16 packs lose nothing extra before the int8 rounding itself.
func quantizeQ8(w weightMat, n, k int) *q8Mat {
	q := &q8Mat{
		data:  make([]int8, n*k),
		scale: make([]float32, n),
	}
	row := make([]float32, k)
	for j := 0; j < n; j++ {
		w.rowInto(row, j, k)
		q.scale[j] = quantizeRowQ8(q.data[j*k:j*k+k], row)
	}
	return q
}

// q8Kernel is the optional Q8 entry point a Kernel may provide. Every
// shipped kernel implements it via matMulTQ8 (pure Go); it exists as a
// separate interface so the experimental path never widens the tested
// Kernel contract.
type q8Kernel interface {
	// MatMulTQ8 computes dst = a · dequant(qb)ᵀ (+ bias) with the scheme
	// documented above. qb is n×k row-major int8, scales has length n.
	MatMulTQ8(dst, a []float32, qb []int8, scales []float32, m, n, k int, bias []float32)
}

// matMulTQ8 is the portable Q8 GEMM all kernels share.
func matMulTQ8(dst, a []float32, qb []int8, scales []float32, m, n, k int, bias []float32, threads int) {
	checkGemmArgs(len(dst), len(a), len(qb), m, n, k, bias)
	if len(scales) < n {
		panic("native: q8 scales smaller than n")
	}
	// Quantize activations per row (dynamic). Serial and cheap relative to
	// the m×n×k GEMM; deterministic regardless of the parallelism below.
	qa := make([]int8, m*k)
	aScale := make([]float32, m)
	for i := 0; i < m; i++ {
		aScale[i] = quantizeRowQ8(qa[i*k:i*k+k], a[i*k:i*k+k])
	}
	parallelRows(m, threads, 1, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			ai := qa[i*k : i*k+k]
			sA := aScale[i]
			di := dst[i*n : i*n+n]
			for j := 0; j < n; j++ {
				bj := qb[j*k : j*k+k]
				var sum int32
				p := 0
				for ; p+4 <= k; p += 4 {
					sum += int32(ai[p])*int32(bj[p]) +
						int32(ai[p+1])*int32(bj[p+1]) +
						int32(ai[p+2])*int32(bj[p+2]) +
						int32(ai[p+3])*int32(bj[p+3])
				}
				for ; p < k; p++ {
					sum += int32(ai[p]) * int32(bj[p])
				}
				s := float32(sum) * (sA * scales[j])
				if bias != nil {
					s += bias[j]
				}
				di[j] = s
			}
		}
	})
}

func (g *goKernel) MatMulTQ8(dst, a []float32, qb []int8, scales []float32, m, n, k int, bias []float32) {
	matMulTQ8(dst, a, qb, scales, m, n, k, bias, g.threads)
}
