package native

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// cosine computes cosine similarity in float64 for test assertions.
func cosine(a, b []float32) float64 {
	if len(a) != len(b) {
		return -2
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return -2
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// loadJSON unmarshals a testdata file into v, fatal on error.
func loadJSON(t *testing.T, path string, v interface{}) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(raw, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

// realWeightsDir returns the pack dir for model id, or "" when the real
// weights are not present on this machine (CI runners: tests must skip).
// FUTON_NATIVE_TEST_MODELS overrides the default ~/.futon/models.
func realWeightsDir(t *testing.T, modelID string) string {
	t.Helper()
	base := os.Getenv("FUTON_NATIVE_TEST_MODELS")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(home, ".futon", "models")
	}
	dir := filepath.Join(base, modelID)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		return ""
	}
	return dir
}

// buildSafetensors constructs an in-memory safetensors image for tests.
// entries maps name → (dtype, shape, raw little-endian payload).
type stTestEntry struct {
	dtype string
	shape []int
	data  []byte
}

func buildSafetensors(t *testing.T, entries map[string]stTestEntry) []byte {
	t.Helper()
	names := make([]string, 0, len(entries))
	for n := range entries {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]interface{}{}
	offset := 0
	var payload []byte
	for _, n := range names {
		e := entries[n]
		header[n] = map[string]interface{}{
			"dtype":        e.dtype,
			"shape":        e.shape,
			"data_offsets": []int{offset, offset + len(e.data)},
		}
		offset += len(e.data)
		payload = append(payload, e.data...)
	}
	hj, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]byte, 8)
	binary.LittleEndian.PutUint64(out, uint64(len(hj)))
	out = append(out, hj...)
	return append(out, payload...)
}

func f32Bytes(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
