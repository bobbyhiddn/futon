// Package packs acquires and verifies native embedding weight packs.
//
// A weight pack is a directory (config.json, model.safetensors, vocab.txt,
// manifest.json, LICENSE) built by scripts/build-weight-pack.py from pinned
// upstream revisions. Packs travel as weight-pack archives (ArchiveSuffix: a
// deterministic tar.zst of the pack directory), published next to a
// ChecksumsAsset file (sha256 per archive) on any static host (see Source).
//
// There are two routes to an installed pack, both landing in the same
// verified state:
//
//  1. Resolve: local-first; downloads from a Source only when the pack is
//     absent and the caller explicitly allowed it.
//  2. Install: fully offline sideload of a weight-pack archive (USB stick,
//     internal mirror). No network, ever.
//
// LOCAL-FIRST IS A CONTRACT, NOT A DEFAULT: when the pack directory is
// already present, Resolve performs zero network activity (pinned by
// TestResolvePresentMakesNoNetworkCalls). Packs live outside the
// application's lifecycle, so a download is a rare, usually once-per-host
// event.
//
// Every failure is loud and typed; nothing partial is ever visible in
// models_dir (extraction happens in a hidden temp dir, then one atomic
// rename). The native provider re-verifies the full manifest at load
// (package native), so verification here is defense in depth, not the only
// gate.
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
	// ChecksumsAsset is the sha256 manifest asset on the weights release.
	ChecksumsAsset = "weights-checksums.txt"

	// ArchiveSuffix is the weight-pack archive extension: a deterministic
	// tar.zst of the pack directory members (flat, no directory prefix).
	// It is the only place the extension is set; Install refuses any other.
	// scripts/build-weight-pack.py and .github/workflows/weights-pack.yml
	// spell it too; TestArchiveSuffixIsSpelledOnce keeps them in step.
	ArchiveSuffix = ".futonw"

	maxChecksumsBytes = int64(1 << 20) // 1 MiB
	maxArchiveBytes   = int64(2 << 30) // 2 GiB compressed
	maxExtractBytes   = int64(4 << 30) // 4 GiB total extracted
	maxArchiveEntries = 64
)

// NotInstalledError: the pack is absent locally and auto-pull is disabled.
type NotInstalledError struct{ PackID string }

func (e *NotInstalledError) Error() string {
	return fmt.Sprintf("packs: pack %q is not installed: install a weight-pack archive (%s) with packs.Install, load a pack directory with packs.LoadDir, or allow a download from a Source", e.PackID, ArchiveSuffix)
}

// UnavailableError: the pack source (or a required asset on it) could
// not be reached or does not exist. Offline hosts land here.
type UnavailableError struct {
	PackID string
	Err    error
}

func (e *UnavailableError) Error() string {
	return fmt.Sprintf("packs: pack %q could not be fetched: %v", e.PackID, e.Err)
}
func (e *UnavailableError) Unwrap() error { return e.Err }

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

// Options configures acquisition.
type Options struct {
	// ModelsDir is the pack root (one subdirectory per pack id).
	ModelsDir string
	// AutoPull permits a network fetch when the pack is absent.
	AutoPull bool
	// Source is where a missing pack is fetched from. Nil means never fetch.
	Source *Source
}

// Resolve returns the installed pack directory for packID, downloading and
// verifying it from the weights release first when absent. Present packs are
// returned with ZERO network activity (manifest presence is checked here;
// the native loader performs the full member-hash verification at load).
func Resolve(modelsDir, packID string) (string, error) {
	return ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true}, packID)
}

// ResolveWith is Resolve with explicit Options.
func ResolveWith(opts Options, packID string) (string, error) {
	if err := validatePackID(packID); err != nil {
		return "", err
	}
	if opts.ModelsDir == "" {
		return "", fmt.Errorf("packs: models dir is required")
	}
	dir := filepath.Join(opts.ModelsDir, packID)
	if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err == nil {
		// Present: local-first contract — no network from here on.
		return dir, nil
	}
	if !opts.AutoPull || opts.Source == nil {
		return "", &NotInstalledError{PackID: packID}
	}
	if err := pull(opts, packID, dir); err != nil {
		return "", err
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

// checksumFor finds the sha256 for name in sha256sum-format data.
func checksumFor(data []byte, name string) (string, error) {
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 2 {
			continue
		}
		candidate := filepath.Base(strings.TrimPrefix(fields[1], "*"))
		if candidate == name {
			sha := strings.ToLower(fields[0])
			if len(sha) != 64 || !isHex(sha) {
				return "", fmt.Errorf("%s entry for %q is not a sha256", ChecksumsAsset, name)
			}
			return sha, nil
		}
	}
	return "", fmt.Errorf("%s has no entry for %q", ChecksumsAsset, name)
}

func isHex(s string) bool {
	_, err := hex.DecodeString(s)
	return err == nil
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
