package native

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// Golden-parity gates against the pinned upstream reference (see
// scripts/gen-native-embed-fixtures.py — fixtures are cross-checked between
// two independent reference implementations before being committed).
//
// parityGateMin is the hard gate: below it the native engine may NOT share a
// vector space with reference-generated vectors. parityFlagMin is the soft
// bar from the verification escalation: any individual text below it is
// reported even when the hard gate passes.
const (
	parityGateMin = 0.999
	parityFlagMin = 0.9995
)

type parityFixture struct {
	ModelID   string `json:"model_id"`
	Revision  string `json:"revision"`
	Reference string `json:"reference"`
	Cases     []struct {
		Name   string    `json:"name"`
		Text   string    `json:"text"`
		Vector []float32 `json:"vector"`
	} `json:"cases"`
	Matryoshka *struct {
		Dim     int         `json:"dim"`
		Vectors [][]float32 `json:"vectors"`
	} `json:"matryoshka"`
}

// loadParityProvider loads the real-weight pack for modelID, skipping the
// test when the weights are absent (CI runners never have them; dev boxes
// build them with scripts/build-weight-pack.py).
func loadParityProvider(t *testing.T, modelID string) (*Provider, parityFixture) {
	t.Helper()
	dir := realWeightsDir(t, modelID)
	if dir == "" {
		t.Skipf("real weights for %s not present (run scripts/build-weight-pack.py); skipping", modelID)
	}
	var fx parityFixture
	loadJSON(t, filepath.Join("testdata", "parity", modelID+".json"), &fx)
	if len(fx.Cases) < 24 {
		t.Fatalf("parity corpus has %d cases, want ≥ 24", len(fx.Cases))
	}
	p, err := New(Options{PackDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Close() })
	return p, fx
}

func runParity(t *testing.T, modelID string) {
	p, fx := loadParityProvider(t, modelID)
	minCos, meanCos := 1.0, 0.0
	worst := ""
	for _, c := range fx.Cases {
		got, err := p.Embed(context.Background(), c.Text)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		cos := cosine(got, c.Vector)
		meanCos += cos
		if cos < minCos {
			minCos, worst = cos, c.Name
		}
		if cos < parityGateMin {
			t.Errorf("PARITY GATE FAILED %s/%s: cosine %.6f < %.3f", modelID, c.Name, cos, parityGateMin)
		} else if cos < parityFlagMin {
			t.Logf("FLAG %s/%s: cosine %.6f below soft bar %.4f (gate still passed)", modelID, c.Name, cos, parityFlagMin)
		}
		t.Logf("parity %s/%s: cosine %.6f", modelID, c.Name, cos)
	}
	meanCos /= float64(len(fx.Cases))
	t.Logf("PARITY SUMMARY %s: min=%.6f (worst=%s) mean=%.6f over %d texts", modelID, minCos, worst, meanCos, len(fx.Cases))
}

func TestParityRealWeightsNomic(t *testing.T) {
	runParity(t, PackNomicEmbedTextV15)
}

func TestParityRealWeightsBGE(t *testing.T) {
	runParity(t, PackBGESmallENV15)
}

func TestParityRealWeightsArctic(t *testing.T) {
	runParity(t, PackArcticEmbedMLong)
}

// TestArcticTrainedContextTruncation pins the trained-context clamp on the
// arctic pack, mirroring TestNomicTrainedContextTruncation: upstream reaches
// 2049..8192 only through dynamic-NTK RoPE scaling, which the engine does
// not implement — so inputs must truncate at the trained 2048, exactly like
// nomic, and never run the (unimplemented, unfixtured) scaled regime.
func TestArcticTrainedContextTruncation(t *testing.T) {
	dir := realWeightsDir(t, PackArcticEmbedMLong)
	if dir == "" {
		t.Skip("real weights absent; skipping")
	}
	p, err := New(Options{PackDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	cfg := p.model.Config()
	if cfg.EffectiveMaxSeqLen() != 2048 {
		t.Fatalf("arctic effective max = %d, want the trained 2048 (declared %d, trained %d)",
			cfg.EffectiveMaxSeqLen(), cfg.MaxSeqLen, cfg.MaxTrainedPositions)
	}

	var sb strings.Builder
	for i := 0; i < 700; i++ {
		sb.WriteString("the quick brown fox jumps over ")
	}
	long := sb.String()
	got, err := p.Embed(context.Background(), long)
	if err != nil {
		t.Fatal(err)
	}
	idsClamped := p.tok.Encode(long, 2048)
	if len(idsClamped) != 2048 {
		t.Fatalf("clamped encode produced %d ids, want exactly 2048", len(idsClamped))
	}
	want, err := p.model.EncodeTokens(context.Background(), idsClamped, p.dims)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("over-length embed differs bitwise at %d from its first-2048-token truncation", i)
		}
	}
}

// TestParityMatryoshkaNomic checks the published truncation recipe
// (parameter-free layernorm → slice → renormalize) at dim 256 against
// reference vectors computed the same way by the pinned upstream model.
func TestParityMatryoshkaNomic(t *testing.T) {
	dir := realWeightsDir(t, PackNomicEmbedTextV15)
	if dir == "" {
		t.Skip("real weights absent; skipping")
	}
	var fx parityFixture
	loadJSON(t, filepath.Join("testdata", "parity", PackNomicEmbedTextV15+".json"), &fx)
	if fx.Matryoshka == nil {
		t.Fatal("fixture has no matryoshka section")
	}
	p, err := New(Options{PackDir: dir, Dimensions: fx.Matryoshka.Dim})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	minCos := 1.0
	for i, c := range fx.Cases {
		got, err := p.Embed(context.Background(), c.Text)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		if len(got) != fx.Matryoshka.Dim {
			t.Fatalf("dim %d, want %d", len(got), fx.Matryoshka.Dim)
		}
		cos := cosine(got, fx.Matryoshka.Vectors[i])
		if cos < minCos {
			minCos = cos
		}
		if cos < parityGateMin {
			t.Errorf("MATRYOSHKA PARITY FAILED %s: cosine %.6f", c.Name, cos)
		}
	}
	t.Logf("PARITY SUMMARY %s matryoshka-%d: min=%.6f over %d texts",
		PackNomicEmbedTextV15, fx.Matryoshka.Dim, minCos, len(fx.Cases))
}

// TestNomicTrainedContextTruncation pins HIGH-2 on the real pack: a
// >2048-token input must embed bit-identically to its first-2048-token
// truncation. The nomic pack declares max_seq_len 8192, but upstream pins
// rotary_scaling_factor: null — positions past the trained 2048 are plain
// RoPE extrapolation, measured at 0.72 cosine vs the daemon at 3202 tokens.
// Until NTK/dynamic scaling exists and is parity-fixtured, the provider
// truncates at 2048 exactly like the daemon truncates.
func TestNomicTrainedContextTruncation(t *testing.T) {
	dir := realWeightsDir(t, PackNomicEmbedTextV15)
	if dir == "" {
		t.Skip("real weights absent; skipping")
	}
	p, err := New(Options{PackDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	cfg := p.model.Config()
	if cfg.EffectiveMaxSeqLen() != 2048 {
		t.Fatalf("nomic effective max = %d, want the trained 2048 (declared %d, trained %d)",
			cfg.EffectiveMaxSeqLen(), cfg.MaxSeqLen, cfg.MaxTrainedPositions)
	}

	var sb strings.Builder
	for i := 0; i < 700; i++ {
		sb.WriteString("the quick brown fox jumps over ")
	}
	long := sb.String()
	if ids := p.tok.Encode(long, cfg.MaxSeqLen); len(ids) <= 2048 {
		t.Fatalf("test input only %d tokens; must exceed the trained context", len(ids))
	}

	got, err := p.Embed(context.Background(), long)
	if err != nil {
		t.Fatal(err)
	}
	idsClamped := p.tok.Encode(long, 2048)
	if len(idsClamped) != 2048 {
		t.Fatalf("clamped encode produced %d ids, want exactly 2048", len(idsClamped))
	}
	want, err := p.model.EncodeTokens(context.Background(), idsClamped, p.dims)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("over-length embed differs bitwise at %d from its first-2048-token truncation", i)
		}
	}
}

// TestRealWeightsBatchMatchesSingle re-pins batch/single bit-equality on the
// real packs (the toy version runs in CI; this one runs wherever weights are).
func TestRealWeightsBatchMatchesSingle(t *testing.T) {
	p, fx := loadParityProvider(t, PackBGESmallENV15)
	texts := make([]string, 0, 8)
	for _, c := range fx.Cases[:8] {
		texts = append(texts, c.Text)
	}
	batch, err := p.EmbedBatch(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i, text := range texts {
		single, err := p.Embed(context.Background(), text)
		if err != nil {
			t.Fatal(err)
		}
		for j := range single {
			if single[j] != batch[i][j] {
				t.Fatalf("text %d: batch differs from single at %d", i, j)
			}
		}
	}
}
