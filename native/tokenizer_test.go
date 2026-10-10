package native

import (
	"path/filepath"
	"testing"
	"unicode/utf8"
)

// testVocabPath is the committed bert-base-uncased vocab so tokenizer
// conformance runs in CI without any weight download. It must be the exact
// file both launch packs ship: build-weight-pack.py refuses to build a pack
// whose upstream vocab.txt hashes differently (expectedVocabSHA256 there and
// here are the same recorded value), and TestCommittedVocabMatchesPacks pins
// this file — and any local real packs — to that hash.
const (
	testVocabPath = "testdata/vocab_bert_uncased.txt"
	// expectedVocabSHA256 mirrors EXPECTED_VOCAB_SHA256 in
	// scripts/build-weight-pack.py. Changing either side alone must fail.
	expectedVocabSHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
)

// TestCommittedVocabMatchesPacks enforces LOW-4: the committed test vocab is
// byte-identical (by sha256) to the vocab the packs are built with. Without
// this, tokenizer conformance in CI could silently test a different vocab
// than production tokenizes with.
func TestCommittedVocabMatchesPacks(t *testing.T) {
	got, err := sha256File(testVocabPath)
	if err != nil {
		t.Fatal(err)
	}
	if got != expectedVocabSHA256 {
		t.Fatalf("committed test vocab sha256 %s != recorded %s — testdata and pack vocab have diverged", got, expectedVocabSHA256)
	}
	for _, modelID := range []string{PackNomicEmbedTextV15, PackBGESmallENV15, PackArcticEmbedMLong} {
		dir := realWeightsDir(t, modelID)
		if dir == "" {
			continue // pack not on this machine; the build script enforces at build time
		}
		got, err := sha256File(filepath.Join(dir, "vocab.txt"))
		if err != nil {
			t.Fatal(err)
		}
		if got != expectedVocabSHA256 {
			t.Fatalf("%s vocab.txt sha256 %s != recorded %s — pack vocab drifted from the committed test vocab", modelID, got, expectedVocabSHA256)
		}
	}
}

func defaultTokCfg() TokenizerConfig {
	return TokenizerConfig{DoLowerCase: true, TokenizeChineseChars: true}
}

type tokFixtures struct {
	Models map[string]struct {
		MaxSeqLen int `json:"max_seq_len"`
		Cases     []struct {
			Name     string `json:"name"`
			Text     string `json:"text"`
			IDs      []int  `json:"ids"`
			IDsMax16 []int  `json:"ids_max16"`
		} `json:"cases"`
	} `json:"models"`
}

// TestTokenizerConformance checks Encode against ids produced by the HF fast
// tokenizer (the one sentence-transformers uses) for the whole corpus,
// including truncation pinning at max_len=16.
func TestTokenizerConformance(t *testing.T) {
	var fx tokFixtures
	loadJSON(t, filepath.Join("testdata", "tokenizer_fixtures.json"), &fx)
	if len(fx.Models) == 0 {
		t.Fatal("no models in tokenizer fixtures")
	}
	tok, err := NewTokenizer(testVocabPath, defaultTokCfg())
	if err != nil {
		t.Fatal(err)
	}
	for modelID, mfx := range fx.Models {
		t.Run(modelID, func(t *testing.T) {
			if len(mfx.Cases) < 24 {
				t.Fatalf("only %d fixture cases; corpus should be ≥ 24", len(mfx.Cases))
			}
			for _, c := range mfx.Cases {
				got := tok.Encode(c.Text, mfx.MaxSeqLen)
				if !equalInts(got, c.IDs) {
					t.Errorf("%s: ids mismatch\n got %v\nwant %v", c.Name, got, c.IDs)
				}
				got16 := tok.Encode(c.Text, 16)
				if !equalInts(got16, c.IDsMax16) {
					t.Errorf("%s: max16 ids mismatch\n got %v\nwant %v", c.Name, got16, c.IDsMax16)
				}
			}
		})
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestTokenizerSpecials(t *testing.T) {
	tok, err := NewTokenizer(testVocabPath, defaultTokCfg())
	if err != nil {
		t.Fatal(err)
	}
	if tok.PadID() != 0 {
		t.Errorf("PadID = %d, want 0", tok.PadID())
	}
	// Empty and whitespace-only input reduce to [CLS][SEP].
	for _, text := range []string{"", "   \t\n "} {
		ids := tok.Encode(text, 512)
		if len(ids) != 2 || ids[0] != 101 || ids[1] != 102 {
			t.Errorf("Encode(%q) = %v, want [101 102]", text, ids)
		}
	}
}

func TestTokenizerMissingVocab(t *testing.T) {
	if _, err := NewTokenizer("testdata/does-not-exist.txt", defaultTokCfg()); err == nil {
		t.Fatal("expected error for missing vocab")
	}
}

// FuzzTokenizerEncode: never panic, and always produce a well-formed
// [CLS] … [SEP] sequence within bounds, for arbitrary byte soup.
func FuzzTokenizerEncode(f *testing.F) {
	tok, err := NewTokenizer(testVocabPath, defaultTokCfg())
	if err != nil {
		f.Fatal(err)
	}
	seeds := []string{
		"", " ", "hello world", "Élodie café ☃", "自然言語処理",
		"(((([[{{}}]]))))", "a\x00b\x07c", "\xff\xfe invalid utf8",
		"buffalo buffalo buffalo", "x", "[CLS] injected [SEP]",
	}
	for _, s := range seeds {
		f.Add(s, uint16(512))
	}
	f.Fuzz(func(t *testing.T, text string, maxLen16 uint16) {
		maxLen := int(maxLen16%1024) + 2
		ids := tok.Encode(text, maxLen)
		if len(ids) < 2 || len(ids) > maxLen {
			t.Fatalf("len(ids)=%d outside [2,%d]", len(ids), maxLen)
		}
		if ids[0] != 101 || ids[len(ids)-1] != 102 {
			t.Fatalf("sequence not [CLS]…[SEP]: %v", ids)
		}
		for _, id := range ids {
			if id < 0 || id >= tok.VocabSize() {
				t.Fatalf("id %d outside vocab %d", id, tok.VocabSize())
			}
		}
		// Determinism under re-encoding.
		again := tok.Encode(text, maxLen)
		if !equalInts(ids, again) {
			t.Fatalf("non-deterministic encode for %q", text)
		}
		_ = utf8.ValidString(text)
	})
}
