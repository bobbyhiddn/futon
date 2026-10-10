package packs

import (
	"archive/tar"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// extractAndVerify unpacks a weight-pack archive into a hidden temp dir inside
// modelsDir (same filesystem — the eventual rename must be atomic) and
// verifies EVERY member against the archived manifest.json. wantPackID ""
// accepts whatever pack the manifest declares (offline install); otherwise
// the manifest's model_id must match. On success the caller owns tmpDir; on
// error it has already been removed — no partial pack is ever visible.
func extractAndVerify(modelsDir, archivePath, wantPackID string) (*packManifest, string, error) {
	label := wantPackID
	if label == "" {
		label = filepath.Base(archivePath)
	}
	if err := os.MkdirAll(modelsDir, 0o700); err != nil {
		return nil, "", fmt.Errorf("packs: create models dir: %w", err)
	}
	tmpDir, err := os.MkdirTemp(modelsDir, ".weights-extract-*")
	if err != nil {
		return nil, "", fmt.Errorf("packs: create extraction temp dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(tmpDir) }

	if err := extractArchive(archivePath, tmpDir); err != nil {
		cleanup()
		return nil, "", &VerificationError{PackID: label, Reason: err.Error()}
	}
	man, err := verifyDir(tmpDir, label)
	if err != nil {
		cleanup()
		return nil, "", err
	}
	if err := validatePackID(man.ModelID); err != nil {
		cleanup()
		return nil, "", &VerificationError{PackID: label, Member: "manifest.json", Reason: fmt.Sprintf("model_id %q is not a valid pack id", man.ModelID)}
	}
	if wantPackID != "" && man.ModelID != wantPackID {
		cleanup()
		return nil, "", &VerificationError{PackID: wantPackID, Member: "manifest.json", Reason: fmt.Sprintf("archive declares model_id %q", man.ModelID)}
	}
	return man, tmpDir, nil
}

// extractArchive unpacks a tar.zst of flat regular files into dst.
// Hostile-archive rules: regular files only, safe flat names, bounded entry
// count and total size. Anything else is a loud refusal.
func extractArchive(archivePath, dst string) error {
	f, err := os.Open(archivePath) // #nosec G304 -- caller-supplied archive path
	if err != nil {
		return err
	}
	defer f.Close()
	zr, err := zstd.NewReader(f)
	if err != nil {
		return fmt.Errorf("not a zstd stream (is this a %s archive?): %w", ArchiveSuffix, err)
	}
	defer zr.Close()

	tr := tar.NewReader(zr)
	entries := 0
	var total int64
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("reading archive: %w", err)
		}
		entries++
		if entries > maxArchiveEntries {
			return fmt.Errorf("archive has more than %d entries", maxArchiveEntries)
		}
		if hdr.Typeflag != tar.TypeReg {
			return fmt.Errorf("archive entry %q is not a regular file (type %q)", hdr.Name, rune(hdr.Typeflag))
		}
		if err := validateMemberName(hdr.Name); err != nil {
			return err
		}
		if hdr.Size < 0 {
			return fmt.Errorf("archive entry %q has negative size", hdr.Name)
		}
		if hdr.Size > maxExtractBytes || total > maxExtractBytes-hdr.Size {
			return fmt.Errorf("archive expands past the %d-byte limit", maxExtractBytes)
		}
		total += hdr.Size
		memberFile, err := memberPath(dst, hdr.Name)
		if err != nil {
			return err
		}
		out, err := os.OpenFile(memberFile, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			return err
		}
		n, err := io.Copy(out, io.LimitReader(tr, hdr.Size+1))
		if closeErr := out.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			return fmt.Errorf("extracting %q: %w", hdr.Name, err)
		}
		if n != hdr.Size {
			return fmt.Errorf("extracting %q: wrote %d bytes, header declares %d", hdr.Name, n, hdr.Size)
		}
	}
	if entries == 0 {
		return errors.New("archive is empty")
	}
	return nil
}

// ErrUnsupportedArchive is wrapped by the refusal of a file that is not
// named as a weight-pack archive (ArchiveSuffix).
var ErrUnsupportedArchive = errors.New("unsupported weight-pack archive")

// CheckArchivePath refuses a path whose extension is not ArchiveSuffix
// (compared case-insensitively). futon installs no other archive name,
func CheckArchivePath(path string) error {
	if strings.EqualFold(filepath.Ext(strings.TrimSpace(path)), ArchiveSuffix) {
		return nil
	}
	return fmt.Errorf("packs: %w: %s is not a weight-pack archive; only %s archives install", ErrUnsupportedArchive, filepath.Base(path), ArchiveSuffix)
}

// Install sideloads a weight-pack archive (ArchiveSuffix) fully offline:
// refuse any other file name, verify every member hash against the archived
// manifest, then atomically install under the manifest's model_id. No
// network is ever touched. A same-id pack with a DIFFERENT manifest is
// refused unless force; an identical pack is a no-op.
func Install(modelsDir, archivePath string, force bool) (string, error) {
	if modelsDir == "" {
		return "", fmt.Errorf("packs: models dir is required")
	}
	if err := CheckArchivePath(archivePath); err != nil {
		return "", err
	}
	man, tmpDir, err := extractAndVerify(modelsDir, archivePath, "")
	if err != nil {
		return "", err
	}
	return installStaged(modelsDir, tmpDir, man.ModelID, force)
}

// VerifyArchive performs the complete hostile-archive and member-integrity
// checks used by Install without installing anything. It is the validation
// boundary used by the upstream catalogue before a weight pack is allowed to leave
// the machine. The returned pack id is the model_id declared by the verified
// archive manifest.
func VerifyArchive(archivePath string) (string, error) {
	modelsDir, err := os.MkdirTemp("", ".futon-weights-verify-*")
	if err != nil {
		return "", fmt.Errorf("packs: create verification directory: %w", err)
	}
	defer os.RemoveAll(modelsDir)

	man, staged, err := extractAndVerify(modelsDir, archivePath, "")
	if err != nil {
		return "", err
	}
	_ = os.RemoveAll(staged)
	return man.ModelID, nil
}

// installStaged is the shared tail of every offline install route (archive
// sideload, directory load): given a fully verified staged pack in tmpDir
// (inside modelsDir), apply the no-clobber contract and commit atomically.
// A same-id pack with a DIFFERENT manifest is refused unless force; an
// identical pack is a no-op. On any error the stage is cleaned up and, for
// a forced replace, the previous pack is restored.
func installStaged(modelsDir, tmpDir, packID string, force bool) (string, error) {
	dir := filepath.Join(modelsDir, packID)

	if _, err := os.Stat(filepath.Join(dir, "manifest.json")); err == nil {
		installedSHA, shaErr := sha256File(filepath.Join(dir, "manifest.json"))
		archiveSHA, shaErr2 := sha256File(filepath.Join(tmpDir, "manifest.json"))
		if shaErr != nil || shaErr2 != nil {
			_ = os.RemoveAll(tmpDir)
			return "", fmt.Errorf("packs: comparing installed pack %q: %v %v", packID, shaErr, shaErr2)
		}
		if installedSHA == archiveSHA {
			// Byte-identical manifest → same pack. Nothing to do.
			_ = os.RemoveAll(tmpDir)
			return packID, nil
		}
		if !force {
			_ = os.RemoveAll(tmpDir)
			return "", &AlreadyInstalledError{PackID: packID, InstalledSHA: installedSHA, ArchiveSHA: archiveSHA}
		}
		// Forced replace: move the old pack aside first so the swap stays
		// atomic from any reader's perspective, then drop it.
		oldDir, err := os.MkdirTemp(modelsDir, ".weights-replaced-*")
		if err != nil {
			_ = os.RemoveAll(tmpDir)
			return "", fmt.Errorf("packs: staging replacement of %q: %w", packID, err)
		}
		_ = os.RemoveAll(oldDir) // want the NAME; rename target must not exist
		if err := os.Rename(dir, oldDir); err != nil {
			_ = os.RemoveAll(tmpDir)
			return "", fmt.Errorf("packs: moving old pack %q aside: %w", packID, err)
		}
		if err := commit(modelsDir, tmpDir, dir, packID); err != nil {
			// Restore the old pack rather than leaving nothing.
			_ = os.Rename(oldDir, dir)
			return "", err
		}
		_ = os.RemoveAll(oldDir)
		return packID, nil
	}

	if err := commit(modelsDir, tmpDir, dir, packID); err != nil {
		return "", err
	}
	return packID, nil
}

// Info describes one installed pack for listing and verification.
type Info struct {
	PackID    string
	Dir       string
	SizeBytes int64  // sum of manifest member sizes + manifest itself
	Revision  string // upstream pin (manifest provenance), may be ""
	Repo      string // upstream repo, may be ""
	License   string
	OK        bool  // full member-hash verification result
	Err       error // verification failure detail when !OK
}

// Verify runs the FULL member-hash check on an installed pack.
func Verify(modelsDir, packID string) (*Info, error) {
	if err := validatePackID(packID); err != nil {
		return nil, err
	}
	dir := filepath.Join(modelsDir, packID)
	if _, err := os.Stat(dir); err != nil {
		return nil, &NotInstalledError{PackID: packID}
	}
	info := &Info{PackID: packID, Dir: dir}
	man, err := verifyDir(dir, packID)
	if err != nil {
		// Best-effort provenance even for a failing pack.
		if m, _, rerr := readManifest(dir); rerr == nil {
			fillProvenance(info, m)
		}
		info.Err = err
		return info, err
	}
	info.OK = true
	fillProvenance(info, man)
	return info, nil
}

func fillProvenance(info *Info, man *packManifest) {
	info.Revision = man.Upstream.Revision
	info.Repo = man.Upstream.Repo
	info.License = man.Upstream.License
	for _, meta := range man.Members {
		info.SizeBytes += meta.Size
	}
	if st, err := os.Stat(filepath.Join(info.Dir, "manifest.json")); err == nil {
		info.SizeBytes += st.Size()
	}
}

// List enumerates installed packs (directories with a manifest.json),
// running the full verification on each. Hidden entries (in-flight
// extractions) and stray files are skipped, not errors.
func List(modelsDir string) ([]Info, error) {
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var infos []Info
	for _, e := range entries {
		name := e.Name()
		if !e.IsDir() || validatePackID(name) != nil {
			continue
		}
		if _, err := os.Stat(filepath.Join(modelsDir, name, "manifest.json")); err != nil {
			continue // not a pack (e.g. unrelated directories share this tree)
		}
		info, err := Verify(modelsDir, name)
		if err != nil && info == nil {
			return nil, err
		}
		infos = append(infos, *info)
	}
	return infos, nil
}
