package futon

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/bobbyhiddn/futon/packs"
)

// toyPackDir points at the committed CI-runnable toy pack so factory/auto
// tests never need real weights.
func toyPackDir() string {
	return filepath.Join("native", "testdata", "toy", "toy-rope-swiglu")
}

func TestNativePackForModel(t *testing.T) {
	cases := map[string]string{
		"nomic-embed-text":              "nomic-embed-text-v1.5",
		"nomic-embed-text:latest":       "nomic-embed-text-v1.5",
		"nomic-embed-text-v1.5":         "nomic-embed-text-v1.5",
		"":                              "nomic-embed-text-v1.5",
		"bge-small-en-v1.5":             "bge-small-en-v1.5",
		"bge-small":                     "bge-small-en-v1.5",
		"snowflake-arctic-embed-m-long": "snowflake-arctic-embed-m-long",
		"snowflake-arctic":              "snowflake-arctic-embed-m-long",
		"all-minilm":                    "",
		"text-embedding-004":            "",
	}
	for model, want := range cases {
		if got := nativePackForModel(model); got != want {
			t.Errorf("nativePackForModel(%q) = %q, want %q", model, got, want)
		}
	}
}

func TestNativeModelDimensions(t *testing.T) {
	tests := map[string]int{
		"":                              768,
		"nomic-embed-text-v1.5":         768,
		"bge-small":                     384,
		"snowflake-arctic-embed-m-long": 768,
	}
	for model, want := range tests {
		got, ok := NativeModelDimensions(model)
		if !ok || got != want {
			t.Errorf("NativeModelDimensions(%q) = (%d, %v), want (%d, true)", model, got, ok, want)
		}
	}
	if got, ok := NativeModelDimensions("unknown-cloud-model"); ok || got != 0 {
		t.Errorf("unknown model dimensions = (%d, %v), want (0, false)", got, ok)
	}
}

func TestNativeThreadBudget(t *testing.T) {
	tests := []struct {
		name                 string
		requested, available int
		want                 int
	}{
		{name: "automatic workstation cap", requested: 0, available: 24, want: 6},
		{name: "automatic small host", requested: 0, available: 4, want: 4},
		{name: "explicit smaller budget", requested: 3, available: 24, want: 3},
		{name: "explicit clamped to host", requested: 64, available: 12, want: 12},
		{name: "invalid availability stays live", requested: 0, available: 0, want: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := nativeThreadBudget(tt.requested, tt.available); got != tt.want {
				t.Fatalf("nativeThreadBudget(%d, %d) = %d, want %d",
					tt.requested, tt.available, got, tt.want)
			}
		})
	}
}

func TestResolvePackDir(t *testing.T) {
	// Explicit ModelPath wins.
	dir, err := resolvePackDir(Config{ModelPath: "/tmp/pack"})
	if err != nil || dir != "/tmp/pack" {
		t.Errorf("ModelPath precedence: got %q, %v", dir, err)
	}
	// ModelsDir + mapped model.
	dir, err = resolvePackDir(Config{ModelsDir: "/models", Model: "nomic-embed-text"})
	if err != nil || dir != filepath.Join("/models", "nomic-embed-text-v1.5") {
		t.Errorf("ModelsDir resolution: got %q, %v", dir, err)
	}
	// Unmapped model without ModelPath is a loud error.
	if _, err := resolvePackDir(Config{Model: "text-embedding-004"}); err == nil {
		t.Error("unmapped model resolved silently")
	}
}

func TestFactoryNativeProvider(t *testing.T) {
	p, err := New(Config{Provider: "native", ModelPath: toyPackDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	// Engine-qualified for provenance: native and daemon vectors are
	// different spaces and must never share a stored name.
	if p.ModelName() != "native/toy-rope-swiglu" {
		t.Errorf("ModelName = %q, want engine-qualified %q", p.ModelName(), "native/toy-rope-swiglu")
	}
	vec, err := p.Embed(context.Background(), "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != p.Dimensions() {
		t.Errorf("vector dim %d != Dimensions() %d", len(vec), p.Dimensions())
	}
	if !p.IsAvailable(context.Background()) {
		t.Error("native provider reports unavailable")
	}
}

// TestAutoDegradesToNullWhenNoPack pins the auto contract: no pack means the
// null provider, with no daemon and no network.
func TestAutoDegradesToNullWhenNoPack(t *testing.T) {
	p, err := NewAutoProvider(Config{
		Provider: "auto",
		Model:    "nomic-embed-text",
		Timeout:  2 * time.Second, //deadline:deliberate the provider config field under test — a wire value, not a test wait
		// ModelPath deliberately unset and no pack in any default dir
		// mapping for this ModelsDir. AutoPull off: this test is about the
		// no-pack fallback, not acquisition (and tests never touch a real
		// upstream).
		ModelsDir: t.TempDir(),
		AutoPull:  boolPtr(false),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if _, ok := p.(*NullProvider); !ok {
		t.Fatalf("auto without a pack returned %T, want *NullProvider", p)
	}
}

func TestEmptyProviderDefaultsToNative(t *testing.T) {
	p, err := New(Config{ModelPath: toyPackDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if !strings.HasPrefix(p.ModelName(), "native/") {
		t.Fatalf("empty provider model = %q, want native provenance", p.ModelName())
	}
}

func boolPtr(b bool) *bool { return &b }

// TestNativeAutoPullDisabledSurfacesTypedError: with auto_pull off and no
// installed pack, construction fails with packs's typed not-installed
// error — loud refusal, zero network.
func TestNativeAutoPullDisabledSurfacesTypedError(t *testing.T) {
	_, err := NewNativeProvider(Config{
		Provider:  "native",
		Model:     "nomic-embed-text",
		ModelsDir: t.TempDir(),
		AutoPull:  boolPtr(false),
	})
	var notInstalled *packs.NotInstalledError
	if !errors.As(err, &notInstalled) {
		t.Fatalf("err = %v, want *packs.NotInstalledError", err)
	}
}

// TestNativeFactoryCallsResolver pins the Phase-2 wiring: the factory routes
// pack acquisition through the packs resolver (which is local-first and
// verified), passing the mapped pack id, the models dir, and the auto_pull
// default of FALSE.
func TestNativeFactoryCallsResolver(t *testing.T) {
	orig := packsResolve
	defer func() { packsResolve = orig }()

	var gotOpts packs.Options
	var gotPack string
	packsResolve = func(opts packs.Options, packID string) (string, error) {
		gotOpts = opts
		gotPack = packID
		// Redirect to the committed toy pack so construction succeeds.
		return toyPackDir(), nil
	}

	modelsDir := t.TempDir()
	p, err := NewNativeProvider(Config{
		Provider:  "native",
		Model:     "nomic-embed-text",
		ModelsDir: modelsDir,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	if gotPack != "nomic-embed-text-v1.5" {
		t.Errorf("resolver pack id = %q", gotPack)
	}
	if gotOpts.ModelsDir != modelsDir {
		t.Errorf("resolver models dir = %q, want %q", gotOpts.ModelsDir, modelsDir)
	}
	if gotOpts.AutoPull {
		t.Error("auto_pull must DEFAULT to FALSE when unset — weight acquisition is an explicit act the host application must opt into")
	}
	if p.ModelName() != "native/toy-rope-swiglu" {
		t.Errorf("ModelName = %q", p.ModelName())
	}
}

// TestNativeModelPathNeverResolves: an explicit model_path is used as-is —
// the resolver (and any possibility of network) is bypassed entirely.
func TestNativeModelPathNeverResolves(t *testing.T) {
	orig := packsResolve
	defer func() { packsResolve = orig }()
	packsResolve = func(opts packs.Options, packID string) (string, error) {
		t.Errorf("resolver called for explicit model_path (pack %q)", packID)
		return "", nil
	}
	p, err := NewNativeProvider(Config{Provider: "native", ModelPath: toyPackDir()})
	if err != nil {
		t.Fatal(err)
	}
	p.Close()
}
