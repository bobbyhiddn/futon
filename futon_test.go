package futon

import (
	"context"
	"errors"
	"math"
	"path/filepath"
	"testing"

	"github.com/bobbyhiddn/futon/packs"
)

// The toy pack is a tiny random-weight model committed with the native tests.
func toyPackDir() string {
	return filepath.Join("native", "testdata", "toy", "toy-rope-swiglu")
}

func TestClampThreads(t *testing.T) {
	tests := []struct {
		name                 string
		requested, available int
		want                 int
	}{
		{"zero picks the default budget", 0, 16, 6},
		{"default is clamped to few CPUs", 0, 2, 2},
		{"explicit smaller", 3, 16, 3},
		{"explicit larger is clamped", 32, 8, 8},
		{"negative picks default", -1, 16, 6},
		{"no CPUs reported still yields one", 0, 0, 1},
	}
	for _, tt := range tests {
		if got := clampThreads(tt.requested, tt.available); got != tt.want {
			t.Errorf("%s: clampThreads(%d, %d) = %d, want %d", tt.name, tt.requested, tt.available, got, tt.want)
		}
	}
}

func TestNewFromModelPath(t *testing.T) {
	e, err := New(Options{ModelPath: toyPackDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	if e.ModelName() != "native/toy-rope-swiglu" {
		t.Errorf("ModelName = %q, want engine-qualified %q", e.ModelName(), "native/toy-rope-swiglu")
	}
	vec, err := e.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != e.Dimensions() {
		t.Errorf("vector dim %d != Dimensions() %d", len(vec), e.Dimensions())
	}
	var norm float64
	for _, v := range vec {
		norm += float64(v) * float64(v)
	}
	if math.Abs(norm-1) > 1e-3 {
		t.Errorf("vector norm^2 = %v, want ~1 (models normalize)", norm)
	}
}

func TestNewBatchMatchesSingle(t *testing.T) {
	e, err := New(Options{ModelPath: toyPackDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	texts := []string{"hello world", "a longer sentence with several words", ""}
	batch, err := e.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range texts {
		single, err := e.Embed(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		if CosineSimilarity(batch[i], single) < 0.999999 {
			t.Errorf("text %d: batch vector differs from single", i)
		}
	}
}

func TestNewMissingPackIsTyped(t *testing.T) {
	_, err := New(Options{ModelsDir: t.TempDir(), Model: "no-such-pack"})
	var notInstalled *packs.NotInstalledError
	if !errors.As(err, &notInstalled) {
		t.Fatalf("err = %v, want *packs.NotInstalledError", err)
	}
	if notInstalled.PackID != "no-such-pack" {
		t.Errorf("PackID = %q", notInstalled.PackID)
	}
}

func TestNewDefaultModelWhenEmpty(t *testing.T) {
	_, err := New(Options{ModelsDir: t.TempDir()})
	var notInstalled *packs.NotInstalledError
	if !errors.As(err, &notInstalled) || notInstalled.PackID != DefaultModel {
		t.Fatalf("err = %v, want NotInstalledError for %q", err, DefaultModel)
	}
}

func TestNewRejectsPathLikeModelNames(t *testing.T) {
	for _, model := range []string{"../escape", `a\b`, "a/b", "."} {
		if _, err := New(Options{ModelsDir: t.TempDir(), Model: model}); err == nil {
			t.Errorf("New accepted model %q", model)
		}
	}
}

func TestDefaultModelsDirHonoursEnv(t *testing.T) {
	t.Setenv("FUTON_HOME", filepath.Join("some", "home"))
	if got, want := DefaultModelsDir(), filepath.Join("some", "home", "models"); got != want {
		t.Errorf("DefaultModelsDir = %q, want %q", got, want)
	}
}

func TestVectorHelpers(t *testing.T) {
	v := []float32{1, -2, 3.5, 0}
	got := Deserialize(Serialize(v))
	if len(got) != len(v) {
		t.Fatalf("round trip length %d, want %d", len(got), len(v))
	}
	for i := range v {
		if got[i] != v[i] {
			t.Errorf("round trip [%d] = %v, want %v", i, got[i], v[i])
		}
	}
	if Deserialize([]byte{1, 2, 3}) != nil {
		t.Error("Deserialize accepted a length that is not a multiple of 4")
	}
	if c := CosineSimilarity([]float32{1, 0}, []float32{1, 0}); c != 1 {
		t.Errorf("identical vectors: cosine %v, want 1", c)
	}
	if c := CosineSimilarity([]float32{1, 0}, []float32{0, 1}); c != 0 {
		t.Errorf("orthogonal vectors: cosine %v, want 0", c)
	}
	if c := CosineSimilarity([]float32{1, 2}, []float32{1}); c != 0 {
		t.Errorf("length mismatch: cosine %v, want 0", c)
	}
}
