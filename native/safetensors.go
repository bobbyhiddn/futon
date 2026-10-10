// Package native runs BERT-family embedding encoders in pure Go (zero CGO)
// from local weight packs. It slots behind embed.Provider without
// importing the embed package (Go's structural interfaces keep the factory
// dependency one-directional: embed → native).
//
// A weight pack is a directory containing:
//
//	config.json       — architecture + tokenizer options (see PackConfig)
//	model.safetensors — tensors in the canonical naming scheme, F16 or F32
//	vocab.txt         — WordPiece vocabulary, one token per line
//	manifest.json     — provenance: upstream repo/revision, per-member sha256
//	LICENSE           — upstream model license
//
// Packs are produced by scripts/build-weight-pack.py from pinned upstream
// Hugging Face revisions. See docs/design/native-embeddings.md.
package native

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/bits"
	"os"
	"sort"
)

// safetensors dtype identifiers this reader accepts. Everything else is a
// loud error: a pack containing e.g. BF16 or quantized tensors was not built
// by our converter and must not be silently misread.
const (
	dtypeF32 = "F32"
	dtypeF16 = "F16"
)

// maxHeaderBytes bounds the JSON header allocation. Real headers for these
// models are a few KB; 16MB is generous while preventing a corrupt length
// prefix from triggering a multi-GB allocation.
const maxHeaderBytes = 16 << 20

// Tensor is one named tensor from a safetensors file.
//
// Exactly one of Data/F16 is populated. In the default (dequantizing) mode
// every tensor lands in Data as float32. In raw mode (LoadSafetensorsRaw)
// F16 tensors keep their half-precision bits in F16 so the model can hold
// weights at half size and let the kernel convert in-register — the F16→F32
// conversion is exact either way, so both modes compute identical values.
type Tensor struct {
	Name  string
	Shape []int
	// Data is row-major float32, len == product(Shape).
	Data []float32
	// F16 is row-major IEEE 754 binary16 bits (raw mode only).
	F16 []uint16
}

// Dims returns the product of the shape (the element count).
func (t *Tensor) Dims() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// stHeaderEntry is one tensor's metadata in the safetensors JSON header.
type stHeaderEntry struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// LoadSafetensors parses path and returns all tensors keyed by name.
//
// Validation is strict and failures are loud: declared shapes must multiply
// out to the declared byte ranges, ranges must sit inside the data section,
// ranges must not overlap, and together they must cover the data section
// exactly. A malformed or truncated file is an error, never a zero tensor.
func LoadSafetensors(path string) (map[string]*Tensor, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- path is the model-pack weights file chosen by the loader (pack directory + model.safetensors); its bytes are hash-verified against manifest.json
	if err != nil {
		return nil, fmt.Errorf("safetensors: read %s: %w", path, err)
	}
	return ParseSafetensors(raw)
}

// LoadSafetensorsRaw is LoadSafetensors but keeps F16 tensors in their
// half-precision bits (Tensor.F16) instead of dequantizing — the model
// loader uses this so a 274MB fp16 pack occupies ~274MB of weight memory,
// not ~547MB of float32.
func LoadSafetensorsRaw(path string) (map[string]*Tensor, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- path is the model-pack weights file chosen by the loader (pack directory + model.safetensors); its bytes are hash-verified against manifest.json
	if err != nil {
		return nil, fmt.Errorf("safetensors: read %s: %w", path, err)
	}
	return parseSafetensors(raw, true)
}

// ParseSafetensors parses an in-memory safetensors image. See LoadSafetensors.
func ParseSafetensors(raw []byte) (map[string]*Tensor, error) {
	return parseSafetensors(raw, false)
}

func parseSafetensors(raw []byte, keepF16 bool) (map[string]*Tensor, error) {
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors: file too small (%d bytes) for header length prefix", len(raw))
	}
	headerLen64 := binary.LittleEndian.Uint64(raw[:8])
	if headerLen64 > maxHeaderBytes {
		return nil, fmt.Errorf("safetensors: header length %d exceeds limit %d", headerLen64, maxHeaderBytes)
	}
	headerLen := int(headerLen64)
	if len(raw) < 8+headerLen {
		return nil, fmt.Errorf("safetensors: declared header length %d exceeds file size %d", headerLen, len(raw))
	}
	headerJSON := raw[8 : 8+headerLen]
	data := raw[8+headerLen:]

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerJSON, &header); err != nil {
		return nil, fmt.Errorf("safetensors: parse header JSON: %w", err)
	}

	type span struct {
		name       string
		begin, end int
	}
	tensors := make(map[string]*Tensor, len(header))
	spans := make([]span, 0, len(header))

	for name, rawEntry := range header {
		if name == "__metadata__" {
			continue
		}
		var entry stHeaderEntry
		if err := json.Unmarshal(rawEntry, &entry); err != nil {
			return nil, fmt.Errorf("safetensors: tensor %q: parse entry: %w", name, err)
		}
		var elemSize int
		switch entry.Dtype {
		case dtypeF32:
			elemSize = 4
		case dtypeF16:
			elemSize = 2
		default:
			return nil, fmt.Errorf("safetensors: tensor %q: unsupported dtype %q (want F32 or F16)", name, entry.Dtype)
		}
		// Shape product with overflow-checked multiplication: a doctored
		// header like shape [2^32, 2^32] would wrap a naive int product to 0
		// and sail through the span check as an "empty" tensor. bits.Mul64
		// with hi != 0 rejects any product past 64 bits, and the explicit
		// bound below rejects anything whose byte size cannot fit in int.
		elems64 := uint64(1)
		for _, d := range entry.Shape {
			if d < 0 {
				return nil, fmt.Errorf("safetensors: tensor %q: negative dimension %d", name, d)
			}
			hi, lo := bits.Mul64(elems64, uint64(d))
			if hi != 0 {
				return nil, fmt.Errorf("safetensors: tensor %q: shape %v element count overflows", name, entry.Shape)
			}
			elems64 = lo
		}
		if elems64 > uint64(math.MaxInt64)/uint64(elemSize) {
			return nil, fmt.Errorf("safetensors: tensor %q: shape %v byte size overflows", name, entry.Shape)
		}
		elems := int(elems64)
		begin, end := entry.DataOffsets[0], entry.DataOffsets[1]
		if begin < 0 || end < begin || end > len(data) {
			return nil, fmt.Errorf("safetensors: tensor %q: offsets [%d,%d) outside data section of %d bytes",
				name, begin, end, len(data))
		}
		if end-begin != elems*elemSize {
			return nil, fmt.Errorf("safetensors: tensor %q: shape %v × %s needs %d bytes, offsets span %d",
				name, entry.Shape, entry.Dtype, elems*elemSize, end-begin)
		}

		buf := data[begin:end]
		shape := make([]int, len(entry.Shape))
		copy(shape, entry.Shape)
		t := &Tensor{Name: name, Shape: shape}
		switch {
		case entry.Dtype == dtypeF32:
			out := make([]float32, elems)
			for i := range out {
				out[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
			}
			t.Data = out
		case keepF16:
			out := make([]uint16, elems)
			for i := range out {
				out[i] = binary.LittleEndian.Uint16(buf[i*2:])
			}
			t.F16 = out
		default:
			out := make([]float32, elems)
			for i := range out {
				out[i] = f16ToF32(binary.LittleEndian.Uint16(buf[i*2:]))
			}
			t.Data = out
		}
		tensors[name] = t
		spans = append(spans, span{name: name, begin: begin, end: end})
	}

	// Spans must tile the data section exactly: no overlaps, no gaps, no
	// trailing garbage. This is what the reference implementation enforces,
	// and it catches truncated or doctored files.
	sort.Slice(spans, func(i, j int) bool { return spans[i].begin < spans[j].begin })
	cursor := 0
	for _, s := range spans {
		if s.begin != cursor {
			return nil, fmt.Errorf("safetensors: tensor %q begins at %d, expected %d (overlap or gap)", s.name, s.begin, cursor)
		}
		cursor = s.end
	}
	if cursor != len(data) {
		return nil, fmt.Errorf("safetensors: data section is %d bytes but tensors cover %d", len(data), cursor)
	}
	return tensors, nil
}

// f32ToF16 converts float32 to IEEE 754 binary16 bits with
// round-to-nearest-even, handling subnormals, overflow, Inf, and NaN.
// (Used by tests today; the inverse of f16ToF32 up to f16 precision.)
func f32ToF16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp32 := int32(bits >> 23 & 0xff)
	mant := bits & 0x7fffff
	exp := exp32 - 127 + 15

	switch {
	case exp32 == 0xff: // Inf / NaN
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 0x1f: // overflow → Inf
		return sign | 0x7c00
	case exp <= 0: // subnormal or zero
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := mant >> shift
		rem := mant & (1<<shift - 1)
		halfway := uint32(1) << (shift - 1)
		if rem > halfway || (rem == halfway && half&1 == 1) {
			half++
		}
		return sign | uint16(half)
	default:
		half := uint16(exp)<<10 | uint16(mant>>13)
		rem := mant & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && half&1 == 1) {
			half++ // may carry into the exponent; that is correct rounding
		}
		return sign | half
	}
}

// f16ToF32 converts an IEEE 754 binary16 value to float32, including
// subnormals, infinities, and NaN.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h) & 0x3ff

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// Subnormal: normalize into f32.
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13) // Inf / NaN
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
	}
}

// tensorF32 returns the tensor's values as float32, dequantizing raw-mode
// F16 tensors (exactly) when necessary.
func tensorF32(t *Tensor) []float32 {
	if t.Data != nil {
		return t.Data
	}
	out := make([]float32, len(t.F16))
	for i, hbits := range t.F16 {
		out[i] = f16ToF32(hbits)
	}
	return out
}
