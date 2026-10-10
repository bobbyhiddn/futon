package futon

import (
	"log/slog"
	"strings"
)

// New constructs the Provider that cfg selects.
//
// Returns a NullProvider on unknown provider name (never returns error for
// that case — allows config forward-compatibility).
// Returns an error only if required fields are missing or the backend
// cannot be initialised (e.g. AWS credentials not found).
func New(cfg Config) (Provider, error) {
	switch strings.ToLower(cfg.Provider) {
	case "native", "":
		return NewNativeProvider(cfg)
	case "auto":
		return NewAutoProvider(cfg)
	case "gemini":
		return NewGeminiProvider(cfg)
	case "openai":
		return NewOpenAIProvider(cfg)
	case "bedrock":
		return NewBedrockProvider(cfg)
	case "null":
		return &NullProvider{}, nil
	default:
		// Forward-compat: unknown provider → NullProvider + log warning.
		slog.Warn("embed: unknown provider, falling back to null", "provider", cfg.Provider)
		return &NullProvider{}, nil
	}
}
