package native

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
)

// PackManifest is the provenance/integrity record written by
// scripts/build-weight-pack.py (and by the fixture generator for toy packs).
// The loader verifies it BEFORE trusting any pack member (LOW-2): a pack
// whose files do not hash to what the manifest recorded — or whose tensor
// set drifted from the recorded one — must fail loudly at load, never run a
// silently different function.
type PackManifest struct {
	PackFormat int    `json:"pack_format"`
	ModelID    string `json:"model_id"`
	Members    map[string]struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"members"`
	// TensorNames is the exact tensor-name set of model.safetensors,
	// optionals included, so deleting an optional tensor (e.g. a bias)
	// cannot silently change the function. Manifests written before this
	// field existed omit it; the sha256 members still cover the bytes.
	TensorNames []string `json:"tensor_names,omitempty"`
}

// manifestMembers are the pack files the loader actually consumes; each must
// be present in the manifest and hash-verified. (LICENSE is provenance-only
// and not required — toy packs do not carry one.)
var manifestMembers = []string{"config.json", "model.safetensors", "vocab.txt"}

// loadAndVerifyManifest reads <dir>/manifest.json and verifies the sha256
// and size of every member the loader consumes.
//
// Cost: the weights file is hashed in full, once, at load — ~1s for the
// 274MB nomic pack on a weak 4-core box, amortized over the process
// lifetime. That is the price of never running tampered or bit-rotted
// weights; see docs/design/native-embeddings.md.
func loadAndVerifyManifest(dir string) (*PackManifest, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json")) // #nosec G304 -- dir is the configured model-pack directory and manifest.json a fixed member name
	if err != nil {
		return nil, fmt.Errorf("pack %s: manifest.json unreadable (refusing to load an unverifiable pack): %w", dir, err)
	}
	var man PackManifest
	if err := json.Unmarshal(raw, &man); err != nil {
		return nil, fmt.Errorf("pack %s: manifest.json malformed: %w", dir, err)
	}
	for _, member := range manifestMembers {
		meta, ok := man.Members[member]
		if !ok {
			return nil, fmt.Errorf("pack %s: manifest records no entry for %s", dir, member)
		}
		path := filepath.Join(dir, member)
		info, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("pack %s: member %s: %w", dir, member, err)
		}
		if info.Size() != meta.Size {
			return nil, fmt.Errorf("pack %s: member %s is %d bytes, manifest records %d", dir, member, info.Size(), meta.Size)
		}
		got, err := sha256File(path)
		if err != nil {
			return nil, fmt.Errorf("pack %s: member %s: %w", dir, member, err)
		}
		if got != meta.SHA256 {
			return nil, fmt.Errorf("pack %s: member %s sha256 %s does not match manifest %s (tampered or corrupt — rebuild the pack)",
				dir, member, got, meta.SHA256)
		}
	}
	return &man, nil
}

// verifyTensorNames compares the loaded tensor-name set against the
// manifest-recorded one (when recorded). This is the guard that makes
// "delete an optional bias, fix up the member hashes" loud instead of a
// silently different function.
func (man *PackManifest) verifyTensorNames(tensors map[string]*Tensor) error {
	if man.TensorNames == nil {
		return nil // pre-tensor_names manifest; sha256 members still cover the bytes
	}
	got := make([]string, 0, len(tensors))
	for name := range tensors {
		got = append(got, name)
	}
	sort.Strings(got)
	want := append([]string(nil), man.TensorNames...)
	sort.Strings(want)
	if len(got) != len(want) {
		return fmt.Errorf("pack %s: %d tensors on disk, manifest records %d — tensor set drifted", man.ModelID, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			return fmt.Errorf("pack %s: tensor set drifted from manifest (first difference: %q vs recorded %q)", man.ModelID, got[i], want[i])
		}
	}
	return nil
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path) // #nosec G304 -- path is the pack directory joined with a fixed manifestMembers name, opened read-only to hash it
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
