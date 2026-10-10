package futon

import (
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/bobbyhiddn/futon/native"
	"github.com/bobbyhiddn/futon/packs"
)

// Vector-space provenance. The same upstream checkpoint served by a different
// engine can produce different vectors: llama.cpp-based servers tokenize BERT
// text non-conformantly (measured min cosine 0.58 against the reference on
// accented and punctuated text), while this engine tokenizes exactly like the
// reference implementation and holds >= 0.999999 on every corpus text. Vectors
// from different engines are therefore DIFFERENT spaces and must never be
// compared or co-mingled under one name. ModelName is engine-qualified
// ("native/<pack id>") so a store can tell them apart.
const (
	// NativeDefaultModel is the default in-process embedding weight pack.
	NativeDefaultModel = native.PackNomicEmbedTextV15
	// NativeDefaultDimensions is the full-width output of the default pack.
	NativeDefaultDimensions = 768

	// NomicParityGate is the minimum per-text cosine against the pinned
	// reference for an engine to be considered conformant (enforced for the
	// native engine by TestParityRealWeights in package native).
	NomicParityGate = 0.999

	// defaultNativeThreadBudget keeps an interactive application responsive
	// while native inference runs: left alone, every provider would take every
	// logical CPU. Explicit configuration may choose a smaller or larger value,
	// bounded by the process's available parallelism.
	defaultNativeThreadBudget = 6
)

// nativeThreadBudget resolves the requested budget against available process
// parallelism. It is kept pure so the policy is pinned without tests mutating
// global GOMAXPROCS state.
func nativeThreadBudget(requested, available int) int {
	if available < 1 {
		available = 1
	}
	if requested < 1 {
		requested = defaultNativeThreadBudget
	}
	if requested > available {
		requested = available
	}
	return requested
}

func resolvedNativeThreads(requested int) int {
	available := runtime.GOMAXPROCS(0)
	if cpus := runtime.NumCPU(); available < 1 || cpus < available {
		available = cpus
	}
	return nativeThreadBudget(requested, available)
}

// nativePackForModel maps a configured model name (a short alias or the
// pack id) to the native pack id, or "" when no native pack exists.
func nativePackForModel(model string) string {
	switch strings.TrimSuffix(strings.TrimSpace(model), ":latest") {
	case "nomic-embed-text", "", native.PackNomicEmbedTextV15:
		return native.PackNomicEmbedTextV15
	case native.PackBGESmallENV15, "bge-small-en", "bge-small":
		return native.PackBGESmallENV15
	case native.PackArcticEmbedMLong, "snowflake-arctic", "arctic-embed-m-long":
		return native.PackArcticEmbedMLong
	default:
		return ""
	}
}

// NativeModelTag returns the engine-qualified provenance tag produced by the
// native provider for a configured model name or known alias.
func NativeModelTag(model string) (string, bool) {
	pack := nativePackForModel(model)
	if pack == "" {
		return "", false
	}
	return native.NativeModelNamePrefix + pack, true
}

// NativeModelDimensions returns the full-width vector size of a known native
// pack or alias. A zero configured Dimensions still lets the pack declare its
// own width; this helper is for registries that need to persist the width
// before the provider is loaded.
func NativeModelDimensions(model string) (int, bool) {
	switch nativePackForModel(model) {
	case native.PackNomicEmbedTextV15, native.PackArcticEmbedMLong:
		return NativeDefaultDimensions, true
	case native.PackBGESmallENV15:
		return 384, true
	default:
		return 0, false
	}
}

// defaultModelsDir resolves where native weight packs live:
// $FUTON_HOME/models when FUTON_HOME is set, else ~/.futon/models.
func defaultModelsDir() string {
	if home := os.Getenv("FUTON_HOME"); home != "" {
		return filepath.Join(home, "models")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".futon", "models")
	}
	return filepath.Join(home, ".futon", "models")
}

// resolvePackDir returns the weight-pack directory for cfg: an explicit
// ModelPath wins; otherwise ModelsDir (or the default) joined with the pack
// id mapped from cfg.Model.
func resolvePackDir(cfg Config) (string, error) {
	if cfg.ModelPath != "" {
		return cfg.ModelPath, nil
	}
	pack := nativePackForModel(cfg.Model)
	if pack == "" {
		return "", fmt.Errorf("native: no weight pack mapping for model %q (set model_path explicitly)", cfg.Model)
	}
	dir := cfg.ModelsDir
	if dir == "" {
		dir = defaultModelsDir()
	}
	packDir := filepath.Join(dir, pack)
	return packDir, nil
}

// autoPullEnabled resolves the tri-state AutoPull config: nil means FALSE.
func autoPullEnabled(cfg Config) bool {
	return cfg.AutoPull != nil && *cfg.AutoPull
}

// packsResolve is the acquisition seam: production always points at
// packs.ResolveWith; tests stub it so no factory test can reach a network.
var packsResolve = func(opts packs.Options, packID string) (string, error) {
	return packs.ResolveWith(opts, packID)
}

// packSource builds the download source from cfg, or nil when none is set.
func packSource(cfg Config) *packs.Source {
	if cfg.PackSourceURL == "" {
		return nil
	}
	return &packs.Source{BaseURL: cfg.PackSourceURL, Token: cfg.PackSourceToken}
}

// NewNativeProvider constructs the pure-Go in-process provider from cfg.
//
// A PRESENT pack is used with zero network activity. A MISSING pack is
// fetched from PackSourceURL only when AutoPull is explicitly true, verified
// end to end, and installed atomically. An explicit ModelPath is always used
// as-is and never downloads. Failed verification or an offline host gets a
// loud typed error; under "auto" that degrades to the null provider.
func NewNativeProvider(cfg Config) (Provider, error) {
	dir, err := resolvePackDir(cfg)
	if err != nil {
		return nil, err
	}
	if cfg.ModelPath == "" {
		if pack := nativePackForModel(cfg.Model); pack != "" {
			modelsDir := cfg.ModelsDir
			if modelsDir == "" {
				modelsDir = defaultModelsDir()
			}
			resolved, resolveErr := packsResolve(packs.Options{
				ModelsDir: modelsDir,
				AutoPull:  autoPullEnabled(cfg),
				Source:    packSource(cfg),
			}, pack)
			if resolveErr != nil {
				return nil, fmt.Errorf("native: %w", resolveErr)
			}
			dir = resolved
		}
	}
	threads := resolvedNativeThreads(cfg.Threads)
	provider, err := native.New(native.Options{
		PackDir:    dir,
		Dimensions: cfg.Dimensions,
		Threads:    threads,
	})
	if err != nil {
		return nil, err
	}
	slog.Info("futon native: provider ready",
		"model", provider.ModelName(), "dimensions", provider.Dimensions(), "threads", threads)
	return provider, nil
}

// NewAutoProvider uses the native in-process engine whenever its weight pack
// is available. If the pack is missing it returns the null provider, so
// callers keep lexical search with no daemon and no network.
func NewAutoProvider(cfg Config) (Provider, error) {
	nat, natErr := NewNativeProvider(cfg)
	if natErr == nil {
		slog.Info("futon auto: using native in-process engine",
			"model", nat.ModelName(), "dims", nat.Dimensions())
		return nat, nil
	}
	slog.Warn("futon auto: native pack unavailable; semantic embedding disabled and lexical search remains active",
		"native_err", natErr)
	return &NullProvider{}, nil
}
