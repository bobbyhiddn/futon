package native

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// TokenizerConfig mirrors the tokenizer options a pack's config.json carries.
// The defaults match HF's BertTokenizerFast for the two launch models
// (uncased, chinese-char isolation, accent stripping implied by lowercasing).
type TokenizerConfig struct {
	// DoLowerCase lowercases and strips accents (the fast BertNormalizer
	// couples the two exactly as HF does: strip_accents defaults to the
	// value of lowercase when unset).
	DoLowerCase bool `json:"do_lower_case"`
	// TokenizeChineseChars isolates CJK ideographs as single tokens.
	TokenizeChineseChars bool `json:"tokenize_chinese_chars"`
	// MaxInputCharsPerWord: words longer than this (in runes) become [UNK]
	// without attempting WordPiece. HF default: 100.
	MaxInputCharsPerWord int `json:"max_input_chars_per_word,omitempty"`

	ClsToken  string `json:"cls_token"`
	SepToken  string `json:"sep_token"`
	UnkToken  string `json:"unk_token"`
	PadToken  string `json:"pad_token"`
	MaskToken string `json:"mask_token"`
}

func (c *TokenizerConfig) applyDefaults() {
	if c.MaxInputCharsPerWord == 0 {
		c.MaxInputCharsPerWord = 100
	}
	if c.ClsToken == "" {
		c.ClsToken = "[CLS]"
	}
	if c.SepToken == "" {
		c.SepToken = "[SEP]"
	}
	if c.UnkToken == "" {
		c.UnkToken = "[UNK]"
	}
	if c.PadToken == "" {
		c.PadToken = "[PAD]"
	}
	if c.MaskToken == "" {
		c.MaskToken = "[MASK]"
	}
}

// Tokenizer implements the HF BERT tokenization pipeline (the fast
// BertNormalizer + BertPreTokenizer + WordPiece path used by
// sentence-transformers): clean text, isolate CJK chars, lowercase +
// NFD accent stripping, punctuation splitting, then greedy longest-match
// WordPiece with "##" continuations.
//
// Literal special-token strings inside input text (a source file containing
// "[CLS]" or "[SEP]") are matched exactly like HF's fast tokenizer matches
// added tokens: case-sensitively, against the RAW text before any
// normalization, splitting the surrounding text into independent segments
// (each segment re-starts word boundaries, so "a[SEP]b" → a, [SEP], b — not
// ##b). Non-special vocab entries like "[unused0]" are NOT literal-matched;
// they split as punctuation, matching HF. callers index source code, which
// contains these literals, so conformance here is pinned by fixtures
// (code_special_literals, special_adversarial).
//
// Safe for concurrent use after construction (all state is read-only).
type Tokenizer struct {
	cfg   TokenizerConfig
	vocab map[string]int

	clsID, sepID, unkID, padID, maskID int

	// specials are the literal added-token strings matched in raw input,
	// with their ids.
	specials []specialLiteral
}

type specialLiteral struct {
	lit string
	id  int
}

// NewTokenizer builds a Tokenizer from a vocab.txt (one token per line,
// line number = token id) and config.
func NewTokenizer(vocabPath string, cfg TokenizerConfig) (*Tokenizer, error) {
	f, err := os.Open(vocabPath) // #nosec G304 -- vocabPath is the model-pack vocab.txt chosen by the loader; its bytes are hash-verified against manifest.json
	if err != nil {
		return nil, fmt.Errorf("tokenizer: open vocab: %w", err)
	}
	defer f.Close()

	cfg.applyDefaults()
	vocab := make(map[string]int, 32768)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)
	id := 0
	for sc.Scan() {
		// HF strips the trailing newline only; interior spaces would be
		// part of the token (none exist in BERT vocabs).
		tok := strings.TrimRight(sc.Text(), "\r\n")
		vocab[tok] = id
		id++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("tokenizer: scan vocab: %w", err)
	}
	if id == 0 {
		return nil, fmt.Errorf("tokenizer: vocab %s is empty", vocabPath)
	}

	t := &Tokenizer{cfg: cfg, vocab: vocab}
	for _, sp := range []struct {
		name string
		dst  *int
	}{
		{cfg.ClsToken, &t.clsID},
		{cfg.SepToken, &t.sepID},
		{cfg.UnkToken, &t.unkID},
		{cfg.PadToken, &t.padID},
		{cfg.MaskToken, &t.maskID},
	} {
		got, ok := vocab[sp.name]
		if !ok {
			return nil, fmt.Errorf("tokenizer: special token %q not in vocab", sp.name)
		}
		*sp.dst = got
		t.specials = append(t.specials, specialLiteral{lit: sp.name, id: got})
	}
	return t, nil
}

// VocabSize returns the number of entries loaded from vocab.txt.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// PadID returns the [PAD] token id.
func (t *Tokenizer) PadID() int { return t.padID }

// Encode tokenizes text and returns token ids wrapped in [CLS] … [SEP],
// truncated so that len(ids) <= maxSeqLen (HF "longest_first" truncation of
// a single sequence: keep the first maxSeqLen-2 content tokens).
func (t *Tokenizer) Encode(text string, maxSeqLen int) []int {
	if maxSeqLen < 2 {
		maxSeqLen = 2
	}
	ids := make([]int, 0, 64)
	ids = append(ids, t.clsID)
	budget := maxSeqLen - 2
encode:
	for _, seg := range t.splitSpecials(text) {
		if seg.specialID >= 0 {
			if len(ids)-1 >= budget {
				break
			}
			ids = append(ids, seg.specialID)
			continue
		}
		for _, w := range t.basicTokenize(seg.text) {
			if len(ids)-1 >= budget {
				break encode
			}
			for _, p := range t.wordPiece(w) {
				if len(ids)-1 >= budget {
					break encode
				}
				ids = append(ids, p)
			}
		}
	}
	ids = append(ids, t.sepID)
	return ids
}

// segment is one span of input: either a literal special-token match
// (specialID >= 0, text empty) or plain text to run through the normalizer
// pipeline (specialID == -1).
type segment struct {
	text      string
	specialID int
}

// splitSpecials splits raw input on literal special-token strings, exactly
// like HF's fast-tokenizer added-token extraction: case-sensitive, on the
// UN-normalized text, non-overlapping leftmost matches. Text on either side
// of a match becomes an independent segment (word boundaries reset — pinned
// by the special_adversarial fixture: "[CLS]x[SEP]" → [CLS], "x", [SEP]).
func (t *Tokenizer) splitSpecials(text string) []segment {
	var segs []segment
	for len(text) > 0 {
		best, bestID, bestLen := -1, -1, 0
		for _, sp := range t.specials {
			// Leftmost match wins; at equal position the longer literal
			// wins (the five defaults can never tie, but configs could).
			if i := strings.Index(text, sp.lit); i >= 0 &&
				(best < 0 || i < best || (i == best && len(sp.lit) > bestLen)) {
				best, bestID, bestLen = i, sp.id, len(sp.lit)
			}
		}
		if best < 0 {
			segs = append(segs, segment{text: text, specialID: -1})
			break
		}
		if best > 0 {
			segs = append(segs, segment{text: text[:best], specialID: -1})
		}
		segs = append(segs, segment{specialID: bestID})
		text = text[best+bestLen:]
	}
	return segs
}

// basicTokenize runs the BertNormalizer + pre-tokenizer pipeline and returns
// whitespace/punctuation-delimited word units ready for WordPiece.
func (t *Tokenizer) basicTokenize(text string) []string {
	var sb strings.Builder
	sb.Grow(len(text) + len(text)/4)

	// Clean: drop NUL/replacement/control chars, canonicalize whitespace.
	// Then optionally isolate CJK ideographs with surrounding spaces.
	for _, r := range text {
		switch {
		case r == 0x0000 || r == 0xFFFD || isControl(r):
			// dropped
		case isWhitespace(r):
			sb.WriteByte(' ')
		case t.cfg.TokenizeChineseChars && isCJK(r):
			sb.WriteByte(' ')
			sb.WriteRune(r)
			sb.WriteByte(' ')
		default:
			sb.WriteRune(r)
		}
	}
	cleaned := sb.String()

	if t.cfg.DoLowerCase {
		cleaned = strings.ToLower(cleaned)
		cleaned = stripAccents(cleaned)
	}

	// Split on whitespace, then split each chunk on punctuation with every
	// punctuation rune becoming its own token.
	var words []string
	for _, chunk := range strings.Fields(cleaned) {
		start := -1
		for i, r := range chunk {
			if isPunct(r) {
				if start >= 0 {
					words = append(words, chunk[start:i])
					start = -1
				}
				words = append(words, string(r))
			} else if start < 0 {
				start = i
			}
		}
		if start >= 0 {
			words = append(words, chunk[start:])
		}
	}
	return words
}

// wordPiece splits one word into vocabulary piece ids by greedy
// longest-match-first, "##"-prefixing continuation pieces. A word that
// cannot be fully segmented (or is longer than MaxInputCharsPerWord runes)
// becomes a single [UNK].
func (t *Tokenizer) wordPiece(word string) []int {
	runes := []rune(word)
	if len(runes) > t.cfg.MaxInputCharsPerWord {
		return []int{t.unkID}
	}
	var out []int
	start := 0
	for start < len(runes) {
		end := len(runes)
		curID := -1
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				curID = id
				break
			}
			end--
		}
		if curID < 0 {
			return []int{t.unkID}
		}
		out = append(out, curID)
		start = end
	}
	return out
}

// stripAccents applies NFD and removes nonspacing marks (category Mn),
// matching HF's strip-accents step.
func stripAccents(s string) string {
	decomposed := norm.NFD.String(s)
	var sb strings.Builder
	sb.Grow(len(decomposed))
	for _, r := range decomposed {
		if unicode.Is(unicode.Mn, r) {
			continue
		}
		sb.WriteRune(r)
	}
	return sb.String()
}

// isWhitespace matches HF _is_whitespace: space, \t, \n, \r, or Unicode Zs.
func isWhitespace(r rune) bool {
	switch r {
	case ' ', '\t', '\n', '\r':
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

// isControl matches HF _is_control: category C* except the whitespace
// characters \t, \n, \r.
func isControl(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	}
	return unicode.Is(unicode.C, r)
}

// isPunct matches HF _is_punctuation: the four ASCII symbol/punct ranges
// (which include $ + < = > ^ ` | ~, not punctuation in Unicode terms) plus
// Unicode category P*.
func isPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.Is(unicode.P, r)
}

// isCJK matches HF _is_chinese_char: the CJK ideograph blocks (deliberately
// not Hangul/Katakana/Hiragana, which have their own word boundaries).
func isCJK(r rune) bool {
	switch {
	case r >= 0x4E00 && r <= 0x9FFF,
		r >= 0x3400 && r <= 0x4DBF,
		r >= 0x20000 && r <= 0x2A6DF,
		r >= 0x2A700 && r <= 0x2B73F,
		r >= 0x2B740 && r <= 0x2B81F,
		r >= 0x2B820 && r <= 0x2CEAF,
		r >= 0xF900 && r <= 0xFAFF,
		r >= 0x2F800 && r <= 0x2FA1F:
		return true
	}
	return false
}
