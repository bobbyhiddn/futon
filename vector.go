package futon

import (
	"encoding/binary"
	"math"
)

// Serialize packs a float32 slice into a little-endian binary BLOB.
// Format: little-endian float32[N] — identical to Python's struct.pack("<Nf", ...).
func Serialize(v []float32) []byte {
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// Deserialize unpacks a little-endian BLOB into a float32 slice.
// Returns nil if input length is not a multiple of 4.
func Deserialize(b []byte) []float32 {
	if len(b) == 0 || len(b)%4 != 0 {
		return nil
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// CosineSimilarity computes cosine similarity of two float32 vectors.
// Returns 0.0 on length mismatch or zero vectors. Clamped to [-1.0, 1.0].
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	sim := dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
	if sim > 1.0 {
		return 1.0
	}
	if sim < -1.0 {
		return -1.0
	}
	return sim
}
