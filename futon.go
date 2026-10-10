// Package futon turns text into vectors. It is the embedding half of a
// retrieval system and nothing more: one Embedder interface and an in-process
// pure-Go engine that runs BERT-family encoders from local weight packs. No
// daemon, no CGO, no network.
//
// All implementations are safe for concurrent use.
//
//	e, err := futon.New(futon.Options{Model: "bge-small-en-v1.5"})
//	if err != nil { ... }
//	defer e.Close()
//	vec, err := e.Embed(ctx, "the quick brown fox")
//
// A model is a weight pack: a directory under ModelsDir named by the model.
// See package packs to install, verify and list packs, and docs/packs.md for
// building packs from Hugging Face models.
package futon

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/bobbyhiddn/futon/native"
	"github.com/bobbyhiddn/futon/packs"
)

// Embedder turns text into vectors.
//
// Implementations are safe for concurrent use and honour ctx cancellation.
type Embedder interface {
	// Embed returns the embedding of one text.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch embeds several texts. The result is index-aligned with texts.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions is the output vector size.
	Dimensions() int

	// ModelName is a stable identifier to store next to every vector. It is
	// engine-qualified ("native/<pack id>"): the same checkpoint served by a
	// different engine can produce different vectors, and vectors from
	// different engines must never be compared.
	ModelName() string

	// Close releases the model's memory. Safe to call more than once.
	Close() error
}

// DefaultModel is the pack used when Options.Model is empty: small, fast and a
// good general-purpose retrieval model.
const DefaultModel = "bge-small-en-v1.5"

// Options selects and configures a model.
type Options struct {
	// Model is the pack id: the name of a directory under ModelsDir. Empty
	// means DefaultModel.
	Model string

	// ModelsDir holds one subdirectory per pack. Empty means
	// $FUTON_HOME/models, falling back to ~/.futon/models.
	ModelsDir string

	// ModelPath points directly at one pack directory, overriding
	// Model/ModelsDir.
	ModelPath string

	// Dimensions, when non-zero, truncates the output of models that support
	// matryoshka embeddings (for example nomic-embed-text-v1.5 at 256).
	// Zero means the model's full width.
	Dimensions int

	// Threads is the CPU thread budget for one embedder. Zero picks
	// min(6, available CPUs) so an interactive application stays responsive;
	// explicit values are clamped to the available CPUs.
	Threads int
}

// New loads the model that opts selects. It fails with *packs.NotInstalledError
// when the pack is not on disk, and with a descriptive error when the pack
// fails verification.
func New(opts Options) (Embedder, error) {
	dir, err := packDir(opts)
	if err != nil {
		return nil, err
	}
	e, err := native.New(native.Options{
		PackDir:    dir,
		Dimensions: opts.Dimensions,
		Threads:    threadBudget(opts.Threads),
	})
	if err != nil {
		return nil, err
	}
	return e, nil
}

// DefaultModelsDir is where packs live when Options.ModelsDir is empty:
// $FUTON_HOME/models, else ~/.futon/models.
func DefaultModelsDir() string {
	if home := os.Getenv("FUTON_HOME"); home != "" {
		return filepath.Join(home, "models")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return filepath.Join(".futon", "models")
	}
	return filepath.Join(home, ".futon", "models")
}

func packDir(opts Options) (string, error) {
	if opts.ModelPath != "" {
		return opts.ModelPath, nil
	}
	model := opts.Model
	if model == "" {
		model = DefaultModel
	}
	modelsDir := opts.ModelsDir
	if modelsDir == "" {
		modelsDir = DefaultModelsDir()
	}
	dir, err := packs.Resolve(modelsDir, model)
	if err != nil {
		return "", fmt.Errorf("futon: %w", err)
	}
	return dir, nil
}

const defaultThreadBudget = 6

// threadBudget resolves the requested budget against available parallelism.
func threadBudget(requested int) int {
	available := runtime.GOMAXPROCS(0)
	if cpus := runtime.NumCPU(); available < 1 || cpus < available {
		available = cpus
	}
	return clampThreads(requested, available)
}

func clampThreads(requested, available int) int {
	if available < 1 {
		available = 1
	}
	if requested < 1 {
		requested = defaultThreadBudget
	}
	if requested > available {
		requested = available
	}
	return requested
}
