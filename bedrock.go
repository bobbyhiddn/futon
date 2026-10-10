package futon

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	BedrockDefaultModel      = "amazon.titan-embed-text-v2:0"
	BedrockDefaultDimensions = 1024
	BedrockDefaultRegion     = "us-east-1"
	bedrockAvailCacheTTL     = 30 * time.Second
)

// BedrockProvider implements Provider using AWS Bedrock Runtime.
//
// Uses stdlib net/http with AWS Signature V4 signing — no external SDK required.
// Sequential only (Bedrock has no native batch embedding endpoint).
//
// Auth: reads AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN from
// the environment. Instance roles and profile-based auth are not supported in
// this implementation — use the AWS SDK variant for those.
type BedrockProvider struct {
	region     string
	model      string
	dimensions int
	client     *http.Client
	accessKey  string
	secretKey  string
	token      string // optional session token

	availMu      sync.Mutex
	availResult  bool
	availChecked time.Time
}

// NewBedrockProvider constructs a BedrockProvider from cfg.
//
// Reads AWS credentials from standard environment variables:
//   - AWS_ACCESS_KEY_ID (required)
//   - AWS_SECRET_ACCESS_KEY (required)
//   - AWS_SESSION_TOKEN (optional)
func NewBedrockProvider(cfg Config) (*BedrockProvider, error) {
	region := cfg.AWSRegion
	if region == "" {
		region = BedrockDefaultRegion
	}
	model := cfg.Model
	if model == "" {
		model = BedrockDefaultModel
	}
	dims := cfg.Dimensions
	if dims == 0 {
		dims = BedrockDefaultDimensions
	}
	timeout := cfg.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	accessKey := os.Getenv("AWS_ACCESS_KEY_ID")
	secretKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	token := os.Getenv("AWS_SESSION_TOKEN")

	if accessKey == "" || secretKey == "" {
		return nil, fmt.Errorf("bedrock: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set")
	}

	return &BedrockProvider{
		region:     region,
		model:      model,
		dimensions: dims,
		client:     &http.Client{Timeout: timeout},
		accessKey:  accessKey,
		secretKey:  secretKey,
		token:      token,
	}, nil
}

// Embed invokes the Bedrock InvokeModel API for a single text embedding.
func (p *BedrockProvider) Embed(ctx context.Context, text string) ([]float32, error) {
	reqBody := map[string]interface{}{
		"inputText":  text,
		"dimensions": p.dimensions,
		"normalize":  true,
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("bedrock: marshal request: %w", err)
	}

	endpoint := fmt.Sprintf("https://bedrock-runtime.%s.amazonaws.com/model/%s/invoke",
		p.region, p.model)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("bedrock: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	if err := bedrockSignAWS4(req, bodyBytes, p.region, "bedrock", p.accessKey, p.secretKey, p.token); err != nil {
		return nil, fmt.Errorf("bedrock: sign request: %w", err)
	}

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("bedrock: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("bedrock: unexpected status %d", resp.StatusCode)
	}

	var data struct {
		Embedding           []float32 `json:"embedding"`
		InputTextTokenCount int       `json:"inputTextTokenCount"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("bedrock: decode response: %w", err)
	}
	if len(data.Embedding) == 0 {
		return nil, fmt.Errorf("bedrock: empty embedding in response")
	}
	return data.Embedding, nil
}

// EmbedBatch generates embeddings sequentially (no native batch endpoint).
func (p *BedrockProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		vec, err := p.Embed(ctx, t)
		if err != nil {
			slog.Warn("bedrock: EmbedBatch item failed", "index", i, "err", err)
			continue
		}
		result[i] = vec
	}
	return result, nil
}

// Dimensions returns the configured vector dimensionality.
func (p *BedrockProvider) Dimensions() int { return p.dimensions }

// ModelName returns the Bedrock model ID.
func (p *BedrockProvider) ModelName() string { return p.model }

// IsAvailable probes with a test embed call. Caches for 30s.
func (p *BedrockProvider) IsAvailable(ctx context.Context) bool {
	p.availMu.Lock()
	defer p.availMu.Unlock()

	if time.Since(p.availChecked) < bedrockAvailCacheTTL {
		return p.availResult
	}

	testCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	_, err := p.Embed(testCtx, "test")
	result := err == nil
	p.availResult = result
	p.availChecked = time.Now()
	return result
}

// Close releases idle HTTP connections.
func (p *BedrockProvider) Close() error {
	p.client.CloseIdleConnections()
	return nil
}

// Compile-time interface check.
var _ Provider = (*BedrockProvider)(nil)

// ── AWS SigV4 signing ──────────────────────────────────────────────────────────

// bedrockSignAWS4 signs an HTTP request with AWS Signature Version 4.
// Minimal implementation for Bedrock InvokeModel — no dependency on aws-sdk-go.
func bedrockSignAWS4(req *http.Request, body []byte, region, service, accessKey, secretKey, token string) error {
	now := time.Now().UTC()
	dateStamp := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")

	req.Header.Set("x-amz-date", amzDate)
	if token != "" {
		req.Header.Set("x-amz-security-token", token)
	}

	// Payload hash
	h := sha256.Sum256(body)
	payloadHash := fmt.Sprintf("%x", h)
	req.Header.Set("x-amz-content-sha256", payloadHash)

	// Canonical headers
	canonicalHeaders, signedHeaders := bedrockCanonicalHeaders(req)

	// Canonical request
	canonicalRequest := strings.Join([]string{
		req.Method,
		req.URL.Path,
		req.URL.RawQuery,
		canonicalHeaders,
		signedHeaders,
		payloadHash,
	}, "\n")

	// String to sign
	credentialScope := strings.Join([]string{dateStamp, region, service, "aws4_request"}, "/")
	h2 := sha256.Sum256([]byte(canonicalRequest))
	stringToSign := strings.Join([]string{
		"AWS4-HMAC-SHA256",
		amzDate,
		credentialScope,
		fmt.Sprintf("%x", h2),
	}, "\n")

	// Signing key
	signingKey := bedrockDeriveKey(secretKey, dateStamp, region, service)

	// Signature
	mac := hmac.New(sha256.New, signingKey)
	mac.Write([]byte(stringToSign))
	signature := fmt.Sprintf("%x", mac.Sum(nil))

	// Authorization header
	auth := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		accessKey, credentialScope, signedHeaders, signature)
	req.Header.Set("Authorization", auth)
	return nil
}

func bedrockCanonicalHeaders(req *http.Request) (canonical, signed string) {
	headers := make(map[string]string)
	var keys []string

	for k, v := range req.Header {
		lk := strings.ToLower(k)
		if _, exists := headers[lk]; !exists {
			headers[lk] = strings.TrimSpace(v[0])
			keys = append(keys, lk)
		}
	}

	// Always include host
	if _, ok := headers["host"]; !ok {
		host := req.Host
		if host == "" {
			host = req.URL.Host
		}
		headers["host"] = host
		keys = append(keys, "host")
	}

	sort.Strings(keys)

	var canonLines []string
	for _, k := range keys {
		canonLines = append(canonLines, k+":"+headers[k])
	}
	canonical = strings.Join(canonLines, "\n") + "\n"
	signed = strings.Join(keys, ";")
	return canonical, signed
}

func bedrockDeriveKey(secretKey, dateStamp, region, service string) []byte {
	hmacSHA256 := func(key []byte, data string) []byte {
		h := hmac.New(sha256.New, key)
		h.Write([]byte(data))
		return h.Sum(nil)
	}
	kDate := hmacSHA256([]byte("AWS4"+secretKey), dateStamp)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	return hmacSHA256(kService, "aws4_request")
}
