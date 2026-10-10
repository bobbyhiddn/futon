package futon

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"
)

const (
	GeminiDefaultModel      = "text-embedding-004"
	GeminiDefaultDimensions = 768
	geminiAPIBase           = "https://generativelanguage.googleapis.com/v1beta/models"
	geminiAvailCacheTTL     = 30 * time.Second
)

// GeminiProvider implements Provider using the Google Gemini embedding API.
//
// Uses pure stdlib net/http — no external SDK required.
// Sequential only (Gemini API has no batch endpoint in v1).
type GeminiProvider struct {
	apiKey     string
	model      string
	dimensions int
	client     *http.Client

	availMu      sync.Mutex
	availResult  bool
	availChecked time.Time
}

// NewGeminiProvider constructs a GeminiProvider from cfg.
func NewGeminiProvider(cfg Config) (*GeminiProvider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("gemini: api_key is required")
	}
	model := cfg.Model
	if model == "" {
		model = GeminiDefaultModel
	}
	dims := cfg.Dimensions
	if dims == 0 {
		dims = GeminiDefaultDimensions
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	return &GeminiProvider{
		apiKey:     cfg.APIKey,
		model:      model,
		dimensions: dims,
		client:     &http.Client{Timeout: timeout},
	}, nil
}

// geminiEmbedRequest is the JSON request body for the Gemini embedContent API.
type geminiEmbedRequest struct {
	Model   string             `json:"model"`
	Content geminiEmbedContent `json:"content"`
}

type geminiEmbedContent struct {
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

// Embed generates a single embedding vector via the Gemini embedContent API.
func (p *GeminiProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	url := fmt.Sprintf("%s/%s:embedContent?key=%s", geminiAPIBase, p.model, p.apiKey)

	reqBody := geminiEmbedRequest{
		Model: "models/" + p.model,
		Content: geminiEmbedContent{
			Parts: []geminiPart{{Text: text}},
		},
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("gemini: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("gemini: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("gemini: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("gemini: unexpected status %d", resp.StatusCode)
	}

	var data struct {
		Embedding struct {
			Values []float32 `json:"values"`
		} `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("gemini: decode response: %w", err)
	}
	if len(data.Embedding.Values) == 0 {
		return nil, fmt.Errorf("gemini: empty embedding in response")
	}
	return data.Embedding.Values, nil
}

// EmbedBatch generates embeddings sequentially (Gemini has no batch endpoint).
func (p *GeminiProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := p.Embed(ctx, t)
		if err != nil {
			slog.Warn("gemini: EmbedBatch item failed", "index", i, "err", err)
			// nil entry — partial failure tolerated
			continue
		}
		result[i] = vec
	}
	return result, nil
}

// Dimensions returns the vector dimensionality.
func (p *GeminiProvider) Dimensions() int { return p.dimensions }

// ModelName returns the Gemini model name.
func (p *GeminiProvider) ModelName() string { return p.model }

// IsAvailable tests reachability via a 1-word probe embed. Caches for 30s.
func (p *GeminiProvider) IsAvailable(ctx context.Context) bool {
	p.availMu.Lock()
	defer p.availMu.Unlock()

	if time.Since(p.availChecked) < geminiAvailCacheTTL {
		return p.availResult
	}

	testCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	_, err := p.Embed(testCtx, "test")
	result := err == nil
	p.availResult = result
	p.availChecked = time.Now()
	return result
}

// Close is a no-op (stdlib HTTP client — idle connections cleaned by GC).
func (p *GeminiProvider) Close() error {
	p.client.CloseIdleConnections()
	return nil
}

// Compile-time interface check.
var _ Provider = (*GeminiProvider)(nil)
