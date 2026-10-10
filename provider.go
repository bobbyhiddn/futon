// Package futon turns text into vectors. It is the embedding half of a
// retrieval system and nothing more: one Provider interface, an in-process
// pure-Go engine that runs BERT-family encoders from local weight packs (no
// daemon, no CGO), and HTTP backends for hosted or OpenAI-compatible servers.
//
// All implementations are safe for concurrent use. Callers invoke Embed/EmbedBatch
// from multiple goroutines during parallel scans.
package futon

import (
	"context"
	"fmt"
	"time"
)

// Provider is the interface all embedding backends must satisfy.
//
// Implementations must be safe for concurrent use. Callers invoke
// Embed/EmbedBatch from multiple goroutines (parallel collection scans).
//
// All methods must honour ctx cancellation. IsAvailable may cache results
// for a short TTL but should not hold locks on the caller goroutine.
type Provider interface {
	// Embed generates a single embedding vector. Returns (nil, nil) for
	// NullProvider (no error, graceful no-op). Returns (nil, err) on real
	// failures — callers should fall back to FTS5 when Embed fails.
	Embed(ctx context.Context, text string) ([]float32, error)

	// EmbedBatch generates embeddings for multiple texts in one call.
	// Implementations use batch API calls when the backend supports them
	// (e.g. an OpenAI-style input:[]string request). Falls back to
	// sequential Embed calls when the batch API is unavailable.
	// Returned slice is index-aligned with texts; nil entries signal
	// per-item failure when partial results are tolerated.
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)

	// Dimensions returns the output vector size. 0 for NullProvider.
	Dimensions() int

	// ModelName returns a stable identifier (e.g. "nomic-embed-text",
	// "text-embedding-004"). Stored alongside each embedding for provenance.
	ModelName() string

	// IsAvailable checks if the backend is reachable and the model is
	// loaded. Safe to call frequently; implementations may cache with a
	// short TTL (≤ 30s). Returns false immediately for NullProvider.
	IsAvailable(ctx context.Context) bool

	// Close releases held resources (HTTP connections, SDK clients).
	// Safe to call multiple times. Returns nil for NullProvider.
	Close() error
}

// Config is the serializable configuration for any embedding backend.
// Fields unused by a particular provider are ignored.
type Config struct {
	// Provider selects the backend: "native", "auto", "openai", "gemini",
	// "bedrock", "null". Empty defaults to "native". "native" runs in-process
	// from a local weight pack; "auto" uses native when a pack is present and
	// the null provider otherwise, so callers keep lexical search.
	Provider string `yaml:"provider" json:"provider"`

	// Model is the embedding model name.
	// Defaults: native="nomic-embed-text-v1.5", gemini="text-embedding-004",
	//           openai="text-embedding-3-small", bedrock="amazon.titan-embed-text-v2:0"
	Model string `yaml:"model" json:"model"`

	// BaseURL overrides the endpoint of an OpenAI-compatible server (a local
	// llama-server, LM Studio, vLLM, ...). Empty uses the hosted default.
	BaseURL string `yaml:"base_url,omitempty" json:"base_url,omitempty"`

	// APIKey is the API key for Gemini or OpenAI. It is never printed: see String.
	APIKey string `yaml:"api_key,omitempty" json:"api_key,omitempty"`

	// Dimensions overrides the output dimensionality detection.
	// Required for Bedrock (provider does not report dimensions).
	// For the native provider a non-zero value requests matryoshka
	// truncation (pack must support it). 0 means the provider default.
	Dimensions int `yaml:"dimensions,omitempty" json:"dimensions,omitempty"`

	// Threads is the native engine's total CPU thread budget for one
	// provider. Zero selects a conservative automatic budget; explicit values
	// are clamped to the process's available parallelism. Ignored by HTTP
	// providers.
	Threads int `yaml:"threads,omitempty" json:"threads,omitempty"`

	// ModelsDir is the directory holding native weight packs, one
	// subdirectory per pack id. Empty means $FUTON_HOME/models, falling
	// back to ~/.futon/models. Used by the "native" and "auto" providers.
	ModelsDir string `yaml:"models_dir,omitempty" json:"models_dir,omitempty"`

	// ModelPath points directly at one weight-pack directory, overriding
	// ModelsDir/Model resolution. Explicit local paths only: this path
	// never downloads weights.
	ModelPath string `yaml:"model_path,omitempty" json:"model_path,omitempty"`

	// PackSourceURL is a directory URL holding weight-pack archives and a
	// weights-checksums.txt (see package packs). It is only used when
	// AutoPull is true and the pack is missing.
	PackSourceURL string `yaml:"pack_source_url,omitempty" json:"pack_source_url,omitempty"`

	// PackSourceToken is an optional bearer token for PackSourceURL.
	PackSourceToken string `yaml:"pack_source_token,omitempty" json:"pack_source_token,omitempty"`

	// AutoPull permits the "native"/"auto" providers to download a MISSING
	// weight pack from PackSourceURL. Nil defaults to FALSE: a silent
	// multi-hundred-MB download on first use is the wrong default, so the
	// host application asks first (see packs.Resolve, packs.Install). A
	// PRESENT pack is always used with zero network activity, and ModelPath
	// never triggers a download regardless of this setting.
	AutoPull *bool `yaml:"auto_pull,omitempty" json:"auto_pull,omitempty"`

	// Timeout is the HTTP call timeout. Default: 30s.
	Timeout time.Duration `yaml:"timeout,omitempty" json:"timeout,omitempty"`

	// AWS-specific (Bedrock only).
	AWSRegion  string `yaml:"aws_region,omitempty" json:"aws_region,omitempty"`   // e.g. "us-east-1"
	AWSProfile string `yaml:"aws_profile,omitempty" json:"aws_profile,omitempty"` // named profile; empty = env/instance role
}

// String formats the config for logs with its secrets masked, so a stray
// %v or %+v can never print a key.
func (c Config) String() string {
	mask := func(s string) string {
		if s == "" {
			return ""
		}
		return "***"
	}
	c.APIKey = mask(c.APIKey)
	c.PackSourceToken = mask(c.PackSourceToken)
	type plain Config // no String method: avoids recursion
	return fmt.Sprintf("%+v", plain(c))
}
