package native

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// PackFormatVersion is the config.json/pack layout version this loader
// understands. Bump only with a loader that handles both.
const PackFormatVersion = 1

// Position-encoding modes.
const (
	PosAbsolute = "absolute" // learned position embeddings added at the input
	PosRoPE     = "rope"     // rotary embeddings applied to q/k in attention
)

// FFN activations.
const (
	ActGELU   = "gelu"   // fc2(gelu_erf(fc1 x)) — vanilla BERT
	ActSwiGLU = "swiglu" // fc2(fc1(x) ⊙ silu(gate x)) — nomic-bert
)

// Pooling modes.
const (
	PoolMean = "mean" // attention-mask mean over token states (nomic)
	PoolCLS  = "cls"  // first-token ([CLS]) state (bge)
)

// MatryoshkaConfig describes optional output truncation for models trained
// with Matryoshka representation learning.
type MatryoshkaConfig struct {
	// LayerNorm: apply a parameter-free layer norm (eps 1e-5, matching
	// F.layer_norm defaults) to the pooled vector BEFORE truncation —
	// the published nomic-embed-text-v1.5 resizing recipe.
	LayerNorm bool `json:"layernorm"`
	// Dims lists the trained truncation points, largest first,
	// informationally; any dim ≤ hidden_size is accepted.
	Dims []int `json:"dims,omitempty"`
}

// PackConfig is the architecture description in a weight pack's config.json.
// One encoder implementation covers every combination declared here; the two
// launch packs exercise:
//
//	bge-small-en-v1.5:      absolute + gelu + post-LN + cls-pool
//	nomic-embed-text-v1.5:  rope     + swiglu + post-LN + mean-pool
//
// Both launch models are post-layernorm (residual-then-normalize). Pre-LN is
// declared here for forward compatibility and rejected at load until an
// architecture needs it — never silently mis-executed.
type PackConfig struct {
	PackFormat int    `json:"pack_format"`
	ModelID    string `json:"model_id"`

	HiddenSize       int     `json:"hidden_size"`
	NumLayers        int     `json:"num_layers"`
	NumHeads         int     `json:"num_heads"`
	IntermediateSize int     `json:"intermediate_size"`
	VocabSize        int     `json:"vocab_size"`
	TypeVocabSize    int     `json:"type_vocab_size"`
	MaxSeqLen        int     `json:"max_seq_len"`
	LayerNormEps     float64 `json:"layer_norm_eps"`

	// PositionEncoding: "absolute" or "rope".
	PositionEncoding string `json:"position_encoding"`
	// RoPETheta is the rotary base (nomic: 1000, NOT the LLaMA-style 10000).
	// Required when PositionEncoding == "rope".
	RoPETheta float64 `json:"rope_theta,omitempty"`
	// MaxTrainedPositions records the trained context. When set (> 0) it
	// CLAMPS the effective sequence length — see EffectiveMaxSeqLen. The
	// upstream nomic config declares 8192 positions but pins
	// rotary_scaling_factor: null, i.e. plain RoPE extrapolation past the
	// trained 2048 — a known-degraded, unfixtured regime (measured 0.72
	// cosine vs the daemon at 3202 tokens). Until NTK/dynamic scaling
	// exists and is parity-fixtured, inputs are truncated to the trained
	// context, exactly like the daemon truncates.
	MaxTrainedPositions int `json:"max_trained_positions,omitempty"`

	// Activation: "gelu" or "swiglu".
	Activation string `json:"activation"`
	// PreNorm selects pre-layernorm blocks. Both launch models are
	// post-norm; true is currently rejected at load (see above).
	PreNorm bool `json:"pre_norm,omitempty"`

	// Pooling: "mean" or "cls".
	Pooling string `json:"pooling"`
	// Normalize: L2-normalize the pooled vector (both launch models).
	Normalize bool `json:"normalize"`

	Matryoshka *MatryoshkaConfig `json:"matryoshka,omitempty"`

	Tokenizer TokenizerConfig `json:"tokenizer"`

	// Prefixes optionally records the upstream task-prefix templates (e.g.
	// nomic's "search_query: "). Provenance only: futon embeds raw text —
	// the provider NEVER applies a prefix. Continuity with vectors already
	// stored by prefix-less engines requires prefix-less embedding.
	Prefixes map[string]string `json:"prefixes,omitempty"`
}

// EffectiveMaxSeqLen is the sequence-length budget the provider and model
// actually honor: the pack's declared max_seq_len clamped to the trained
// context (max_trained_positions) when one is declared. Running positions
// the model never saw in training (plain RoPE extrapolation) is a silent
// quality cliff, so the clamp is enforced at tokenization (head truncation,
// [SEP] preserved — identical semantics to the declared-max truncation and
// to the daemon's) and re-checked by EncodeTokens.
func (c *PackConfig) EffectiveMaxSeqLen() int {
	if c.MaxTrainedPositions > 0 && c.MaxTrainedPositions < c.MaxSeqLen {
		return c.MaxTrainedPositions
	}
	return c.MaxSeqLen
}

// Validate checks internal consistency; load fails loudly on any violation.
func (c *PackConfig) Validate() error {
	if c.PackFormat != PackFormatVersion {
		return fmt.Errorf("pack: unsupported pack_format %d (loader supports %d)", c.PackFormat, PackFormatVersion)
	}
	if c.ModelID == "" {
		return fmt.Errorf("pack: model_id missing")
	}
	if c.HiddenSize <= 0 || c.NumLayers <= 0 || c.NumHeads <= 0 || c.IntermediateSize <= 0 {
		return fmt.Errorf("pack %s: non-positive architecture dimension", c.ModelID)
	}
	if c.HiddenSize%c.NumHeads != 0 {
		return fmt.Errorf("pack %s: hidden_size %d not divisible by num_heads %d", c.ModelID, c.HiddenSize, c.NumHeads)
	}
	if c.VocabSize <= 0 || c.MaxSeqLen <= 2 {
		return fmt.Errorf("pack %s: invalid vocab_size/max_seq_len", c.ModelID)
	}
	if c.MaxTrainedPositions < 0 || (c.MaxTrainedPositions > 0 && c.MaxTrainedPositions <= 2) {
		return fmt.Errorf("pack %s: invalid max_trained_positions %d", c.ModelID, c.MaxTrainedPositions)
	}
	if c.LayerNormEps <= 0 {
		return fmt.Errorf("pack %s: layer_norm_eps must be positive", c.ModelID)
	}
	switch c.PositionEncoding {
	case PosAbsolute:
	case PosRoPE:
		if c.RoPETheta <= 0 {
			return fmt.Errorf("pack %s: rope position encoding requires rope_theta > 0", c.ModelID)
		}
	default:
		return fmt.Errorf("pack %s: unknown position_encoding %q", c.ModelID, c.PositionEncoding)
	}
	switch c.Activation {
	case ActGELU, ActSwiGLU:
	default:
		return fmt.Errorf("pack %s: unknown activation %q", c.ModelID, c.Activation)
	}
	switch c.Pooling {
	case PoolMean, PoolCLS:
	default:
		return fmt.Errorf("pack %s: unknown pooling %q", c.ModelID, c.Pooling)
	}
	if c.PreNorm {
		return fmt.Errorf("pack %s: pre_norm is declared but not yet implemented — refusing to run the wrong architecture", c.ModelID)
	}
	if c.Matryoshka != nil {
		for _, d := range c.Matryoshka.Dims {
			if d <= 0 || d > c.HiddenSize {
				return fmt.Errorf("pack %s: matryoshka dim %d outside (0,%d]", c.ModelID, d, c.HiddenSize)
			}
		}
	}
	return nil
}

// LoadPackConfig reads and validates <dir>/config.json.
func LoadPackConfig(dir string) (*PackConfig, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json")) // #nosec G304 -- dir is the configured model-pack directory and config.json a fixed member name
	if err != nil {
		return nil, fmt.Errorf("pack: read config: %w", err)
	}
	var cfg PackConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("pack: parse config: %w", err)
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}
