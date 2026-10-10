package packs

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// LoadDir sideloads a pack DIRECTORY fully offline — the third local route
// beside `pull` (network) and `Install` (weight-pack archive). It exists for packs
// that never became archives: one built straight into a working directory by
// scripts/build-weight-pack.py, or a pack dir copied host-to-host.
//
// Only the members listed in the source's manifest.json (plus the manifest
// itself) are staged — stray files in the source directory are ignored, so a
// scratch build dir can't smuggle anything into models_dir. Every staged
// member is hash-verified against the manifest DURING the copy; the pack is
// then committed under the manifest's model_id with the same atomic
// no-clobber contract as Install (identical pack → no-op; differing pack →
// AlreadyInstalledError unless force, with restore-on-failure).
func LoadDir(modelsDir, srcDir string, force bool) (string, error) {
	if modelsDir == "" {
		return "", fmt.Errorf("packs: models dir is required")
	}
	src, err := filepath.Abs(srcDir)
	if err != nil {
		return "", fmt.Errorf("packs: resolving source dir: %w", err)
	}
	man, rawManifest, err := readManifest(src)
	if err != nil {
		return "", &VerificationError{PackID: filepath.Base(src), Member: "manifest.json", Reason: err.Error()}
	}
	if err := validatePackID(man.ModelID); err != nil {
		return "", &VerificationError{PackID: filepath.Base(src), Member: "manifest.json", Reason: fmt.Sprintf("model_id %q is not a valid pack id", man.ModelID)}
	}
	packID := man.ModelID

	// Refuse loading a pack out of its own installed location: the no-op /
	// replace semantics below assume source and destination are distinct.
	if installed, err := filepath.Abs(filepath.Join(modelsDir, packID)); err == nil && installed == src {
		return "", fmt.Errorf("packs: %q is already the installed location of pack %q", srcDir, packID)
	}

	for _, req := range requiredMembers {
		if _, ok := man.Members[req]; !ok {
			return "", &VerificationError{PackID: packID, Member: req, Reason: "required member missing from manifest"}
		}
	}

	tmpDir, err := os.MkdirTemp(modelsDir, ".weights-stage-*")
	if err != nil {
		return "", fmt.Errorf("packs: staging dir for %q: %w", packID, err)
	}
	cleanup := func() { _ = os.RemoveAll(tmpDir) }

	for name, want := range man.Members {
		if err := validateMemberName(name); err != nil {
			cleanup()
			return "", &VerificationError{PackID: packID, Member: name, Reason: err.Error()}
		}
		gotSHA, gotSize, err := copyAndHash(src, tmpDir, name)
		if err != nil {
			cleanup()
			return "", &VerificationError{PackID: packID, Member: name, Reason: err.Error()}
		}
		if gotSize != want.Size {
			cleanup()
			return "", &VerificationError{PackID: packID, Member: name,
				Reason: fmt.Sprintf("member %s is %d bytes, manifest records %d", name, gotSize, want.Size)}
		}
		if gotSHA != want.SHA256 {
			cleanup()
			return "", &VerificationError{PackID: packID, Member: name,
				Reason: fmt.Sprintf("member %s sha256 %s, manifest records %s", name, gotSHA, want.SHA256)}
		}
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "manifest.json"), rawManifest, 0o600); err != nil {
		cleanup()
		return "", fmt.Errorf("packs: staging manifest for %q: %w", packID, err)
	}

	return installStaged(modelsDir, tmpDir, packID, force)
}

// copyAndHash streams member name from srcDir into dstDir, returning the
// sha256 hex and byte count of what was actually written — the verification
// hashes the staged copy, never the (still writable) source. Names are
// validated flat, so they cannot resolve outside either directory.
func copyAndHash(srcDir, dstDir, name string) (string, int64, error) {
	srcPath, err := memberPath(srcDir, name)
	if err != nil {
		return "", 0, err
	}
	dstPath, err := memberPath(dstDir, name)
	if err != nil {
		return "", 0, err
	}
	in, err := os.Open(srcPath)
	if err != nil {
		return "", 0, err
	}
	defer in.Close()
	out, err := os.OpenFile(dstPath, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return "", 0, err
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(out, h), in)
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}
