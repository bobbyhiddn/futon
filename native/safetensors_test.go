package native

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

func TestSafetensorsRoundTrip(t *testing.T) {
	img := buildSafetensors(t, map[string]stTestEntry{
		"b.mat": {dtype: "F32", shape: []int{2, 3}, data: f32Bytes(1, 2, 3, 4, 5, 6)},
		"a.vec": {dtype: "F32", shape: []int{4}, data: f32Bytes(-1.5, 0, 2.25, 1e-8)},
		"c.f16": {dtype: "F16", shape: []int{3}, data: []byte{
			0x00, 0x3c, // 1.0
			0x00, 0xc0, // -2.0
			0xff, 0x7b, // 65504 (max f16)
		}},
	})
	tensors, err := ParseSafetensors(img)
	if err != nil {
		t.Fatal(err)
	}
	if len(tensors) != 3 {
		t.Fatalf("got %d tensors, want 3", len(tensors))
	}
	mat := tensors["b.mat"]
	if mat.Shape[0] != 2 || mat.Shape[1] != 3 || mat.Data[5] != 6 {
		t.Errorf("b.mat mangled: %+v", mat)
	}
	vec := tensors["a.vec"]
	if vec.Data[0] != -1.5 || vec.Data[3] != 1e-8 {
		t.Errorf("a.vec mangled: %+v", vec.Data)
	}
	f16 := tensors["c.f16"]
	if f16.Data[0] != 1.0 || f16.Data[1] != -2.0 || f16.Data[2] != 65504 {
		t.Errorf("c.f16 dequant wrong: %+v", f16.Data)
	}
}

func TestSafetensorsMalformed(t *testing.T) {
	good := func() map[string]stTestEntry {
		return map[string]stTestEntry{
			"x": {dtype: "F32", shape: []int{2}, data: f32Bytes(1, 2)},
		}
	}

	cases := []struct {
		name    string
		mutate  func(t *testing.T) []byte
		wantSub string
	}{
		{"too_small", func(t *testing.T) []byte { return []byte{1, 2, 3} }, "too small"},
		{"header_past_eof", func(t *testing.T) []byte {
			img := buildSafetensors(t, good())
			binary.LittleEndian.PutUint64(img[:8], uint64(len(img)))
			return img
		}, "exceeds file size"},
		{"header_huge", func(t *testing.T) []byte {
			img := buildSafetensors(t, good())
			binary.LittleEndian.PutUint64(img[:8], 1<<40)
			return img
		}, "exceeds limit"},
		{"bad_dtype", func(t *testing.T) []byte {
			return buildSafetensors(t, map[string]stTestEntry{
				"x": {dtype: "BF16", shape: []int{2}, data: []byte{0, 0, 0, 0}},
			})
		}, "unsupported dtype"},
		{"shape_size_mismatch", func(t *testing.T) []byte {
			return buildSafetensors(t, map[string]stTestEntry{
				"x": {dtype: "F32", shape: []int{3}, data: f32Bytes(1, 2)},
			})
		}, "needs"},
		{"truncated_data", func(t *testing.T) []byte {
			img := buildSafetensors(t, good())
			return img[:len(img)-4]
		}, "outside data section"},
		{"trailing_garbage", func(t *testing.T) []byte {
			img := buildSafetensors(t, good())
			return append(img, 0xde, 0xad)
		}, "tensors cover"},
		{"gap_between_tensors", func(t *testing.T) []byte {
			// Handcraft a header whose second tensor starts past the first's end.
			hdr := `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},` +
				`"b":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			img = append(img, hdr...)
			return append(img, make([]byte, 12)...)
		}, "overlap or gap"},
		{"overlapping_tensors", func(t *testing.T) []byte {
			hdr := `{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},` +
				`"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			img = append(img, hdr...)
			return append(img, make([]byte, 12)...)
		}, "overlap or gap"},
		{"negative_dim", func(t *testing.T) []byte {
			hdr := `{"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}}`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			img = append(img, hdr...)
			return append(img, make([]byte, 8)...)
		}, "negative dimension"},
		{"not_json", func(t *testing.T) []byte {
			hdr := `this is not json`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			return append(img, hdr...)
		}, "parse header"},
		// LOW-1: shape [2^32, 2^32] multiplies to 2^64, which wraps a naive
		// int product to 0 — an empty data span would then pass every check
		// and the tensor would silently load as zero elements. The checked
		// multiply must reject it (the reviewer's exact repro).
		{"shape_product_overflow", func(t *testing.T) []byte {
			hdr := `{"a":{"dtype":"F16","shape":[4294967296,4294967296],"data_offsets":[0,0]}}`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			return append(img, hdr...)
		}, "overflows"},
		// A partial-overflow variant that stays under 2^64 but cannot fit a
		// byte size in int64.
		{"shape_bytes_overflow", func(t *testing.T) []byte {
			hdr := `{"a":{"dtype":"F32","shape":[4294967296,1073741824],"data_offsets":[0,0]}}`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			return append(img, hdr...)
		}, "overflows"},
		// Zero-dim tensors are legal only with an exactly-empty span; a
		// zero dim with a nonzero span is an inconsistency, not a tensor.
		{"zero_dim_nonzero_span", func(t *testing.T) []byte {
			hdr := `{"a":{"dtype":"F32","shape":[0,4],"data_offsets":[0,8]}}`
			img := make([]byte, 8)
			binary.LittleEndian.PutUint64(img, uint64(len(hdr)))
			img = append(img, hdr...)
			return append(img, make([]byte, 8)...)
		}, "needs"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ParseSafetensors(tc.mutate(t))
			if err == nil {
				t.Fatalf("malformed input accepted")
			}
			if !strings.Contains(err.Error(), tc.wantSub) {
				t.Errorf("error %q does not mention %q", err, tc.wantSub)
			}
		})
	}
}

func TestF16Conversion(t *testing.T) {
	cases := []struct {
		bits uint16
		want float32
	}{
		{0x0000, 0},
		{0x8000, float32(math.Copysign(0, -1))},
		{0x3c00, 1},
		{0xbc00, -1},
		{0x4170, 2.71875},
		{0x0001, 5.960464477539063e-08}, // smallest subnormal
		{0x03ff, 6.097555160522461e-05}, // largest subnormal
		{0x7bff, 65504},                 // max finite
	}
	for _, tc := range cases {
		if got := f16ToF32(tc.bits); got != tc.want {
			t.Errorf("f16ToF32(%#04x) = %v, want %v", tc.bits, got, tc.want)
		}
	}
	if got := f16ToF32(0x7c00); !math.IsInf(float64(got), 1) {
		t.Errorf("f16ToF32(+inf bits) = %v", got)
	}
	if got := f16ToF32(0xfc00); !math.IsInf(float64(got), -1) {
		t.Errorf("f16ToF32(-inf bits) = %v", got)
	}
	if got := f16ToF32(0x7e00); !math.IsNaN(float64(got)) {
		t.Errorf("f16ToF32(nan bits) = %v", got)
	}
}

// TestF16RoundTrip: f32→f16→f32 must be the identity for values exactly
// representable in binary16, and f32ToF16 must round-to-nearest-even.
func TestF16RoundTrip(t *testing.T) {
	exact := []float32{0, 1, -1, 0.5, 2.71875, 65504, -65504, 6.097555160522461e-05, 5.960464477539063e-08}
	for _, v := range exact {
		if got := f16ToF32(f32ToF16(v)); got != v {
			t.Errorf("round trip %v → %v", v, got)
		}
	}
	if f16ToF32(f32ToF16(65520)) != float32(math.Inf(1)) {
		t.Error("overflow must saturate to +Inf")
	}
	if !math.IsNaN(float64(f16ToF32(f32ToF16(float32(math.NaN()))))) {
		t.Error("NaN must survive")
	}
	// Nearest-even: 1.0 + 2^-11 is exactly halfway between 1.0 and 1.0+2^-10.
	if got := f16ToF32(f32ToF16(1.0 + 1.0/2048)); got != 1.0 {
		t.Errorf("halfway rounds to even: got %v, want 1.0", got)
	}
	// 1.0+3/2048 is halfway between mantissa 1 (odd) and 2 (even): rounds UP.
	if got := f16ToF32(f32ToF16(1.0 + 3.0/2048)); float64(got) != 1.0+4.0/2048 {
		t.Errorf("halfway rounds to even (odd neighbor case): got %v", got)
	}
}
