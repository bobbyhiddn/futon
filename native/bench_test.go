package native

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Benchmarks embed the same committed corpus the parity fixtures use, so the
// numbers describe realistic chunks (prose, code, logs, unicode).
// Real weights required; benchmarks skip cleanly when absent.

func benchCorpus(b *testing.B, modelID string) []string {
	b.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "parity", modelID+".json"))
	if err != nil {
		b.Skipf("no parity fixture for %s: %v", modelID, err)
	}
	var fx parityFixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		b.Fatal(err)
	}
	texts := make([]string, 0, len(fx.Cases))
	for _, c := range fx.Cases {
		texts = append(texts, c.Text)
	}
	return texts
}

func benchWeights(b *testing.B, modelID string) string {
	b.Helper()
	base := os.Getenv("FUTON_NATIVE_TEST_MODELS")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			b.Skip("no home dir")
		}
		base = filepath.Join(home, ".futon", "models")
	}
	dir := filepath.Join(base, modelID)
	if _, err := os.Stat(filepath.Join(dir, "model.safetensors")); err != nil {
		b.Skipf("real weights for %s not present; skipping benchmark", modelID)
	}
	return dir
}

func countTokens(p *Provider, texts []string) int {
	total := 0
	for _, t := range texts {
		total += len(p.tok.Encode(t, p.model.Config().MaxSeqLen))
	}
	return total
}

func benchNative(b *testing.B, modelID string, threads int) {
	dir := benchWeights(b, modelID)
	texts := benchCorpus(b, modelID)
	p, err := New(Options{PackDir: dir, Threads: threads})
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	if threads == 1 {
		// Constrain batch fan-out too, so "1 thread" means one core total.
		p.batchWorkers = 1
	}
	tokens := countTokens(p, texts)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := p.EmbedBatch(context.Background(), texts); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	perIter := b.Elapsed().Seconds() / float64(b.N)
	b.ReportMetric(float64(tokens)/perIter, "tok/s")
	b.ReportMetric(perIter*1000/float64(len(texts)), "ms/text")
}

func BenchmarkNativeNomic1Thread(b *testing.B) { benchNative(b, PackNomicEmbedTextV15, 1) }
func BenchmarkNativeNomicAllCores(b *testing.B) {
	benchNative(b, PackNomicEmbedTextV15, runtime.NumCPU())
}
func BenchmarkNativeBGE1Thread(b *testing.B)  { benchNative(b, PackBGESmallENV15, 1) }
func BenchmarkNativeBGEAllCores(b *testing.B) { benchNative(b, PackBGESmallENV15, runtime.NumCPU()) }
func BenchmarkNativeArctic1Thread(b *testing.B) {
	benchNative(b, PackArcticEmbedMLong, 1)
}
func BenchmarkNativeArcticAllCores(b *testing.B) {
	benchNative(b, PackArcticEmbedMLong, runtime.NumCPU())
}
