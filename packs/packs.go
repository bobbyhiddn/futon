// Package packs installs, verifies and lists native embedding weight packs.
//
// A weight pack is a directory (config.json, model.safetensors, vocab.txt,
// manifest.json, LICENSE) built by scripts/build-weight-pack.py from a
// Hugging Face model. Packs move between machines as weight-pack archives
// (ArchiveSuffix: a deterministic tar.zst of the pack directory).
//
// There are two ways to put a pack on disk, both landing in the same verified
// state:
//
//  1. Install: unpack a weight-pack archive.
//  2. LoadDir: copy and verify a pack directory (what the build script
//     produces).
//
// This package never touches the network. Every failure is loud and typed,
// and nothing partial is ever visible in models_dir: extraction happens in a
// hidden temp dir, then one atomic rename. The native engine re-verifies the
// full manifest at load (package native), so verification here is defense in
// depth, not the only gate.
package packs

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

const (
	// ArchiveSuffix is the weight-pack archive extension: a deterministic
	// tar.zst of the pack directory members (flat, no directory prefix).
	// It is the only place the extension is set; Install refuses any other.
	// scripts/build-weight-pack.py and .github/workflows/weights-pack.yml
	// spell it too; TestArchiveSuffixIsSpelledOnce keeps them in step.
	ArchiveSuffix = ".futonw"

	maxExtractBytes   = int64(4 << 30) // 4 GiB total extracted
	maxArchiveEntries = 64
)

// NotInstalledError: the pack is absent locally and auto-pull is disabled.
type NotInstalledError struct{ PackID string }

func (e *NotInstalledError) Error() string {
	return fmt.Sprintf("packs: pack %q is not installed: install a weight-pack archive (%s) with packs.Install, or load a pack directory with packs.LoadDir", e.PackID, ArchiveSuffix)
}

// VerificationError: bytes were obtained but failed integrity verification.
// Fail closed — nothing verified is nothing installed.
type VerificationError struct {
	PackID string
	Member string // "" = the archive itself
	Reason string
}

func (e *VerificationError) Error() string {
	if e.Member == "" {
		return fmt.Sprintf("packs: pack %q failed verification: %s (refusing to install)", e.PackID, e.Reason)
	}
	return fmt.Sprintf("packs: pack %q member %q failed verification: %s (refusing to install)", e.PackID, e.Member, e.Reason)
}

// AlreadyInstalledError: an offline Install found a same-id pack whose
// manifest differs from the archive's. Overwriting weights silently is how
// vector spaces drift; the caller must pass force explicitly.
type AlreadyInstalledError struct {
	PackID       string
	InstalledSHA string // sha256 of the installed manifest.json
	ArchiveSHA   string // sha256 of the archive's manifest.json
}

func (e *AlreadyInstalledError) Error() string {
	return fmt.Sprintf("packs: pack %q is already installed with a DIFFERENT manifest (installed manifest sha256 %.12s…, archive %.12s…) — re-run with --force to replace it", e.PackID, e.InstalledSHA, e.ArchiveSHA)
}

// Resolve returns the installed pack directory for packID, or
// *NotInstalledError when it is absent. Manifest presence is checked here; the
// native loader performs the full member-hash verification at load.
func Resolve(modelsDir, packID string) (string, error) {
	if err := validatePackID(packID); err != nil {
		return "", err
	}
	if modelsDir == "" {
		return "", fmt.Errorf("packs: models dir is required")
	}
	dir := filepath.Join(modelsDir, packID)
	if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err != nil {
		return "", &NotInstalledError{PackID: packID}
	}
	return dir, nil
}

// ValidatePackID rejects anything that could escape models_dir — a bare
// directory name only. Exported so CLI surfaces can validate caller input
// BEFORE building any filesystem path from it (gosec G703).
func ValidatePackID(packID string) error {
	return validatePackID(packID)
}

// validatePackID rejects anything that could escape models_dir.
func validatePackID(packID string) error {
	if packID == "" || packID != filepath.Clean(packID) ||
		strings.ContainsAny(packID, `/\`) || packID == "." || packID == ".." ||
		strings.HasPrefix(packID, ".") {
		return fmt.Errorf("packs: invalid pack id %q", packID)
	}
	return nil
}

// commit atomically moves a fully verified extraction into place. A
// concurrent installer winning the race is success, not failure.
func commit(modelsDir, tmpDir, dir, packID string) error {
	if err := os.Rename(tmpDir, dir); err != nil {
		if _, statErr := os.Stat(filepath.Join(dir, "manifest.json")); statErr == nil {
			// Someone else installed it while we worked; ours was verified,
			// theirs will be verified at load. Keep the winner.
			_ = os.RemoveAll(tmpDir)
			return nil
		}
		_ = os.RemoveAll(tmpDir)
		return fmt.Errorf("packs: installing pack %q: %w", packID, err)
	}
	if err := syncDir(modelsDir); err != nil {
		return fmt.Errorf("packs: sync models dir: %w", err)
	}
	return nil
}

// packManifest is the subset of manifest.json this package consumes. It is
// deliberately its own struct: internal/embed/native owns load-time
// verification; this package owns install-time verification and provenance
// display, which additionally needs the upstream block.
type packManifest struct {
	PackFormat int    `json:"pack_format"`
	ModelID    string `json:"model_id"`
	Members    map[string]struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	} `json:"members"`
	Upstream struct {
		Repo     string `json:"repo"`
		Revision string `json:"revision"`
		License  string `json:"license"`
	} `json:"upstream"`
}

// requiredMembers must exist in every installable pack. LICENSE is also
// shipped but toy/test packs may omit it; the manifest governs.
var requiredMembers = []string{"config.json", "model.safetensors", "vocab.txt"}

func readManifest(dir string) (*packManifest, []byte, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "manifest.json")) // #nosec G304 -- dir is the caller-supplied pack source, an installed pack or this package's staging dir; manifest.json is a fixed name
	if err != nil {
		return nil, nil, fmt.Errorf("manifest.json unreadable: %w", err)
	}
	var man packManifest
	if err := json.Unmarshal(raw, &man); err != nil {
		return nil, nil, fmt.Errorf("manifest.json malformed: %w", err)
	}
	if len(man.Members) == 0 {
		return nil, nil, errors.New("manifest.json records no members")
	}
	for _, m := range requiredMembers {
		if _, ok := man.Members[m]; !ok {
			return nil, nil, fmt.Errorf("manifest.json records no entry for required member %s", m)
		}
	}
	return &man, raw, nil
}

// verifyDir verifies EVERY manifest member on disk (sha256 + size) and that
// the directory contains nothing the manifest does not vouch for.
func verifyDir(dir, packID string) (*packManifest, error) {
	man, _, err := readManifest(dir)
	if err != nil {
		return nil, &VerificationError{PackID: packID, Member: "manifest.json", Reason: err.Error()}
	}
	for member, meta := range man.Members {
		if err := validateMemberName(member); err != nil {
			return nil, &VerificationError{PackID: packID, Member: member, Reason: err.Error()}
		}
		path := filepath.Join(dir, member)
		info, err := os.Stat(path)
		if err != nil {
			return nil, &VerificationError{PackID: packID, Member: member, Reason: err.Error()}
		}
		if info.Size() != meta.Size {
			return nil, &VerificationError{PackID: packID, Member: member, Reason: fmt.Sprintf("size %d, manifest records %d", info.Size(), meta.Size)}
		}
		got, err := sha256File(path)
		if err != nil {
			return nil, &VerificationError{PackID: packID, Member: member, Reason: err.Error()}
		}
		if got != meta.SHA256 {
			return nil, &VerificationError{PackID: packID, Member: member, Reason: fmt.Sprintf("sha256 %s does not match manifest %s", got, meta.SHA256)}
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, &VerificationError{PackID: packID, Reason: err.Error()}
	}
	for _, e := range entries {
		name := e.Name()
		if name == "manifest.json" {
			continue
		}
		if _, ok := man.Members[name]; !ok {
			return nil, &VerificationError{PackID: packID, Member: name, Reason: "present on disk but not recorded in manifest.json"}
		}
	}
	return man, nil
}

func validateMemberName(name string) error {
	if name == "" || name != filepath.Clean(name) || strings.ContainsAny(name, `/\:`) ||
		name == "." || name == ".." || strings.HasPrefix(name, ".") {
		return fmt.Errorf("unsafe member name %q", name)
	}
	return nil
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path) // #nosec G304 -- verified pack members under models_dir
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

// memberPath joins a validated flat member name onto dir. Names are validated
// first (no separators, no drive or stream colons, no leading dot), so the
// result can never leave dir.
func memberPath(dir, name string) (string, error) {
	if err := validateMemberName(name); err != nil {
		return "", err
	}
	return filepath.Join(dir, name), nil
}
