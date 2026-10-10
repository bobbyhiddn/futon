package futon

import "context"

// NullProvider is a no-op fallback. Used when no embedding provider is
// configured or when a provider is unavailable and degraded operation is
// required. The recall system detects nil embeddings and falls back to
// FTS5-only search.
type NullProvider struct{}

// Embed always returns (nil, nil) — no embedding, no error.
func (n *NullProvider) Embed(_ context.Context, _ string) ([]float32, error) {
	return nil, nil
}

// EmbedBatch returns a slice of nil vectors — no embeddings, no error.
func (n *NullProvider) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}

// Dimensions returns 0 — NullProvider has no vector space.
func (n *NullProvider) Dimensions() int { return 0 }

// ModelName returns "null".
func (n *NullProvider) ModelName() string { return "null" }

// IsAvailable always returns false.
func (n *NullProvider) IsAvailable(_ context.Context) bool { return false }

// Close is a no-op.
func (n *NullProvider) Close() error { return nil }

// Compile-time interface check.
var _ Provider = (*NullProvider)(nil)
