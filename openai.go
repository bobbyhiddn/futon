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
	OpenAIDefaultModel      = "text-embedding-3-small"
	OpenAIDefaultDimensions = 1536
	OpenAIAPIBase           = "https://api.openai.com/v1"
	openaiAvailCacheTTL     = 30 * time.Second
)

// OpenAIProvider implements Provider using the OpenAI embeddings API.
//
// Uses pure stdlib net/http — no external SDK. Supports native batch embedding
// (OpenAI accepts []string as input in a single request).
type OpenAIProvider struct {
	apiKey     string
	model      string
	dimensions int
	client     *http.Client

	availMu      sync.Mutex
	availResult  bool
	availChecked time.Time
}

// NewOpenAIProvider constructs an OpenAIProvider from cfg.
func NewOpenAIProvider(cfg Config) (*OpenAIProvider, error) {
	if cfg.APIKey == "" {
		return nil, fmt.Errorf("openai: api_key is required")
	}
	model := cfg.Model
	if model == "" {
		model = OpenAIDefaultModel
	}
	dims := cfg.Dimensions
	if dims == 0 {
		switch model {
		case "text-embedding-3-large":
			dims = 3072
		default:
			dims = OpenAIDefaultDimensions
		}
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}
	return &OpenAIProvider{
		apiKey:     cfg.APIKey,
		model:      model,
		dimensions: dims,
		client:     &http.Client{Timeout: timeout},
	}, nil
}

// Embed generates a single embedding vector via the OpenAI embeddings API.
func (p *OpenAIProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := p.embedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) == 0 || vecs[0] == nil {
		return nil, fmt.Errorf("openai: empty embedding response")
	}
	return vecs[0], nil
}

// EmbedBatch generates embeddings for multiple texts in a single API call.
func (p *OpenAIProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	return p.embedBatch(ctx, texts)
}

// embedBatch is the shared implementation that sends a batch to OpenAI.
func (p *OpenAIProvider) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	reqBody := map[string]interface{}{
		"model": p.model,
		"input": texts,
	}
	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		OpenAIAPIBase+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("openai: unexpected status %d", resp.StatusCode)
	}

	var data struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}

	// Build index-aligned result
	result := make([][]float32, len(texts))
	for _, item := range data.Data {
		if item.Index < len(result) {
			result[item.Index] = item.Embedding
		}
	}
	return result, nil
}

// Dimensions returns the vector dimensionality.
func (p *OpenAIProvider) Dimensions() int { return p.dimensions }

// ModelName returns the OpenAI model name.
func (p *OpenAIProvider) ModelName() string { return p.model }

// IsAvailable checks reachability via GET /v1/models. Caches for 30s.
func (p *OpenAIProvider) IsAvailable(ctx context.Context) bool {
	p.availMu.Lock()
	defer p.availMu.Unlock()

	if time.Since(p.availChecked) < openaiAvailCacheTTL {
		return p.availResult
	}

	result := p.checkAvailable(ctx)
	p.availResult = result
	p.availChecked = time.Now()
	return result
}

func (p *OpenAIProvider) checkAvailable(ctx context.Context) bool {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, OpenAIAPIBase+"/models", nil)
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.client.Do(req)
	if err != nil {
		slog.Debug("openai: availability check failed", "err", err)
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// Close releases idle HTTP connections.
func (p *OpenAIProvider) Close() error {
	p.client.CloseIdleConnections()
	return nil
}

// Compile-time interface check.
var _ Provider = (*OpenAIProvider)(nil)
