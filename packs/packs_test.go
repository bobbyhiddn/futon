package packs

import (
	"archive/tar"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
)

// ── fixtures ──────────────────────────────────────────────────────────────────

// fakePackFiles are the non-manifest members of a minimal syntactically-valid
// pack. Content is arbitrary: packs verifies hashes, not model semantics.
func fakePackFiles() map[string][]byte {
	return map[string][]byte{
		"config.json":       []byte(`{"model_id":"fake-pack-v1"}`),
		"model.safetensors": bytes.Repeat([]byte{0xAB}, 4096),
		"vocab.txt":         []byte("[PAD]\n[CLS]\n[SEP]\n[UNK]\nhello\nworld\n"),
		"LICENSE":           []byte("test license\n"),
	}
}

// packBytes renders files + a generated manifest.json into member map form.
func packBytes(t *testing.T, packID string, files map[string][]byte) map[string][]byte {
	t.Helper()
	members := map[string]map[string]any{}
	for name, data := range files {
		sum := sha256.Sum256(data)
		members[name] = map[string]any{"sha256": hex.EncodeToString(sum[:]), "size": len(data)}
	}
	manifest, err := json.MarshalIndent(map[string]any{
		"pack_format": 1,
		"model_id":    packID,
		"members":     members,
		"upstream": map[string]any{
			"repo":     "fixture/repo",
			"revision": "deadbeefcafe",
			"license":  "apache-2.0",
		},
	}, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	out := map[string][]byte{"manifest.json": manifest}
	for k, v := range files {
		out[k] = v
	}
	return out
}

// writePackArchive serializes members as a tar.zst weight-pack archive.
func writePackArchive(t *testing.T, members map[string][]byte) []byte {
	t.Helper()
	var buf bytes.Buffer
	zw, err := zstd.NewWriter(&buf)
	if err != nil {
		t.Fatal(err)
	}
	tw := tar.NewWriter(zw)
	names := make([]string, 0, len(members))
	for name := range members {
		names = append(names, name)
	}
	// Order is irrelevant to the reader; keep it stable anyway.
	for i := 0; i < len(names); i++ {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	for _, name := range names {
		data := members[name]
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data))}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write(data); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func installFakePack(t *testing.T, modelsDir, packID string) {
	t.Helper()
	dir := filepath.Join(modelsDir, packID)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	for name, data := range packBytes(t, packID, fakePackFiles()) {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func checksumsFor(archives map[string][]byte) []byte {
	var b strings.Builder
	for name, data := range archives {
		sum := sha256.Sum256(data)
		fmt.Fprintf(&b, "%s  %s\n", hex.EncodeToString(sum[:]), name)
	}
	return []byte(b.String())
}

// visibleEntries returns non-hidden names in dir — what a reader of
// models_dir can actually see. Atomicity assertions run on this.
func visibleEntries(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), ".") {
			names = append(names, e.Name())
		}
	}
	return names
}

// ── local-first contract ──────────────────────────────────────────────────────

func TestResolvePresentMakesNoNetworkCalls(t *testing.T) {
	modelsDir := t.TempDir()
	installFakePack(t, modelsDir, "fake-pack-v1")

	// ANY request to this server is a contract violation.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("local-first contract violated: network call to %s while pack is present", r.URL)
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()

	dir, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, "fake-pack-v1")
	if err != nil {
		t.Fatal(err)
	}
	if dir != filepath.Join(modelsDir, "fake-pack-v1") {
		t.Fatalf("dir = %q", dir)
	}
}

// TestResolveRealPackPresent exercises the present-path against the real
// local pack store when one exists (built by scripts/build-weight-pack.py).
func TestResolveRealPackPresent(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	modelsDir := filepath.Join(home, ".futon", "models")
	packID := "bge-small-en-v1.5"
	if _, err := os.Stat(filepath.Join(modelsDir, packID, "manifest.json")); err != nil {
		t.Skipf("real pack %s not installed locally", packID)
	}
	dir, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: false}, packID)
	if err != nil {
		t.Fatal(err)
	}
	// And the full verification passes on the real artifact.
	info, err := Verify(modelsDir, packID)
	if err != nil {
		t.Fatalf("real pack failed full verification: %v", err)
	}
	if !info.OK || info.Dir != dir || info.Revision == "" {
		t.Fatalf("info = %+v", info)
	}
}

// ── download path ─────────────────────────────────────────────────────────────

func TestResolveDownloadsVerifiesInstalls(t *testing.T) {
	packID := "fake-pack-v1"
	packArchive := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	archives := map[string][]byte{packID + ArchiveSuffix: packArchive}
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         checksumsFor(archives),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()

	dir, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(modelsDir, packID); err != nil {
		t.Fatalf("installed pack fails verification: %v", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 1 || got[0] != packID {
		t.Fatalf("models dir entries = %v, want just the pack", got)
	}
	// Temp files must be cleaned up.
	entries, _ := os.ReadDir(modelsDir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".weights-") {
			t.Errorf("leftover temp entry %s", e.Name())
		}
	}
	// Second resolve is pure local.
	srv.Close()
	dir2, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	if err != nil || dir2 != dir {
		t.Fatalf("second resolve: %q, %v", dir2, err)
	}
}

func TestResolveAutoPullDisabled(t *testing.T) {
	_, err := ResolveWith(Options{ModelsDir: t.TempDir(), AutoPull: false}, "fake-pack-v1")
	var notInstalled *NotInstalledError
	if !errors.As(err, &notInstalled) {
		t.Fatalf("err = %v, want NotInstalledError", err)
	}
}

func TestResolveOffline(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	source := sourceFor(srv)
	srv.Close() // dead endpoint = offline host
	modelsDir := t.TempDir()
	_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: source}, "fake-pack-v1")
	var unavailable *UnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("err = %v, want UnavailableError", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("offline failure left entries: %v", got)
	}
}

func TestResolveArchiveShaMismatch(t *testing.T) {
	packID := "fake-pack-v1"
	packArchive := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	// Checksums manifest vouches for DIFFERENT bytes.
	bogus := bytes.Repeat([]byte{0x01}, 32)
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         []byte(hex.EncodeToString(bogus) + "  " + packID + ArchiveSuffix + "\n"),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()
	_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("failed pull left entries: %v", got)
	}
}

func TestResolveTamperedMemberInsideArchive(t *testing.T) {
	packID := "fake-pack-v1"
	members := packBytes(t, packID, fakePackFiles())
	// Tamper AFTER manifest generation: outer archive sha will be honest
	// about the tampered bytes, so only member verification can catch it.
	members["model.safetensors"] = bytes.Repeat([]byte{0xEE}, 4096)
	packArchive := writePackArchive(t, members)
	archives := map[string][]byte{packID + ArchiveSuffix: packArchive}
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         checksumsFor(archives),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()
	_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
	if verr.Member != "model.safetensors" {
		t.Errorf("Member = %q, want model.safetensors", verr.Member)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("tampered pull left entries: %v", got)
	}
}

func TestResolveChecksumsMissingEntry(t *testing.T) {
	packID := "fake-pack-v1"
	packArchive := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         []byte("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  other" + ArchiveSuffix + "\n"),
	}
	srv := fakeSource(t, assets)
	_, err := ResolveWith(Options{ModelsDir: t.TempDir(), AutoPull: true, Source: sourceFor(srv)}, packID)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError (fail closed on unvouched asset)", err)
	}
}

func TestResolveUnsafeArchiveEntry(t *testing.T) {
	packID := "fake-pack-v1"
	members := packBytes(t, packID, fakePackFiles())
	members["../evil"] = []byte("escape")
	packArchive := writePackArchive(t, members)
	archives := map[string][]byte{packID + ArchiveSuffix: packArchive}
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         checksumsFor(archives),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()
	_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
	if _, statErr := os.Stat(filepath.Join(filepath.Dir(modelsDir), "evil")); statErr == nil {
		t.Fatal("path traversal escaped models dir")
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("unsafe archive left entries: %v", got)
	}
}

func TestResolveRejectsBadPackID(t *testing.T) {
	for _, id := range []string{"", "..", "a/b", `a\b`, ".hidden"} {
		if _, err := Resolve(t.TempDir(), id); err == nil {
			t.Errorf("pack id %q accepted", id)
		}
	}
}

// ── offline install ───────────────────────────────────────────────────────────

func TestInstallOffline(t *testing.T) {
	packID := "fake-pack-v1"
	archivePath := filepath.Join(t.TempDir(), packID+ArchiveSuffix)
	if err := os.WriteFile(archivePath, writePackArchive(t, packBytes(t, packID, fakePackFiles())), 0o600); err != nil {
		t.Fatal(err)
	}
	modelsDir := t.TempDir()

	got, err := Install(modelsDir, archivePath, false)
	if err != nil || got != packID {
		t.Fatalf("Install = %q, %v", got, err)
	}
	if _, err := Verify(modelsDir, packID); err != nil {
		t.Fatalf("sideloaded pack fails verification: %v", err)
	}

	// Identical re-install is a no-op.
	if _, err := Install(modelsDir, archivePath, false); err != nil {
		t.Fatalf("idempotent reinstall: %v", err)
	}

	// Same id, different content: refuse without force.
	files2 := fakePackFiles()
	files2["vocab.txt"] = []byte("[PAD]\n[CLS]\n[SEP]\n[UNK]\ndifferent\n")
	archivePath2 := filepath.Join(t.TempDir(), packID+ArchiveSuffix)
	if err := os.WriteFile(archivePath2, writePackArchive(t, packBytes(t, packID, files2)), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = Install(modelsDir, archivePath2, false)
	var conflict *AlreadyInstalledError
	if !errors.As(err, &conflict) {
		t.Fatalf("err = %v, want AlreadyInstalledError", err)
	}
	// Original still intact.
	if _, err := Verify(modelsDir, packID); err != nil {
		t.Fatalf("refused install disturbed the existing pack: %v", err)
	}

	// Forced replace succeeds and the new content verifies.
	if _, err := Install(modelsDir, archivePath2, true); err != nil {
		t.Fatalf("forced install: %v", err)
	}
	data, err := os.ReadFile(filepath.Join(modelsDir, packID, "vocab.txt"))
	if err != nil || !bytes.Contains(data, []byte("different")) {
		t.Fatalf("forced install did not replace content: %v", err)
	}
	if _, err := Verify(modelsDir, packID); err != nil {
		t.Fatalf("replaced pack fails verification: %v", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 1 {
		t.Fatalf("models dir entries after replace = %v", got)
	}
}

// TestInstallRefusesOtherArchiveNames: only ArchiveSuffix (.futonw) installs.
// A byte-valid pack under the parent-era extension, or any other name, is
// refused before it is opened, and nothing lands in the models dir.
func TestInstallRefusesOtherArchiveNames(t *testing.T) {
	if ArchiveSuffix != ".futonw" {
		t.Fatalf("ArchiveSuffix = %q; the weight-pack extension is .futonw", ArchiveSuffix)
	}
	packID := "fake-pack-v1"
	valid := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	for _, name := range []string{packID + ".hmw", packID + ".HMW", packID + ".zip", packID + ".tar.zst", packID, packID + ".futonw.bak"} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), name)
			if err := os.WriteFile(path, valid, 0o600); err != nil {
				t.Fatal(err)
			}
			modelsDir := t.TempDir()
			_, err := Install(modelsDir, path, false)
			if !errors.Is(err, ErrUnsupportedArchive) {
				t.Fatalf("Install(%s) err = %v, want ErrUnsupportedArchive", name, err)
			}
			if !strings.Contains(err.Error(), ".futonw") {
				t.Errorf("refusal %q does not name the accepted extension", err)
			}
			if got, _ := os.ReadDir(modelsDir); len(got) != 0 {
				t.Fatalf("refused install left entries: %v", got)
			}
		})
	}
	// The same bytes under the uppercase name install, whatever the case.
	for _, name := range []string{packID + ".futonw", packID + ".FUTONW"} {
		path := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(path, valid, 0o600); err != nil {
			t.Fatal(err)
		}
		if got, err := Install(t.TempDir(), path, false); err != nil || got != packID {
			t.Fatalf("Install(%s) = %q, %v", name, got, err)
		}
	}
}

func TestInstallTamperedArchive(t *testing.T) {
	packID := "fake-pack-v1"
	members := packBytes(t, packID, fakePackFiles())
	members["config.json"] = []byte(`{"model_id":"tampered"}`)
	archivePath := filepath.Join(t.TempDir(), packID+ArchiveSuffix)
	if err := os.WriteFile(archivePath, writePackArchive(t, members), 0o600); err != nil {
		t.Fatal(err)
	}
	modelsDir := t.TempDir()
	_, err := Install(modelsDir, archivePath, false)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("tampered install left entries: %v", got)
	}
}

// ── verify / list ─────────────────────────────────────────────────────────────

func TestVerifyDetectsTamperAndStrays(t *testing.T) {
	modelsDir := t.TempDir()
	installFakePack(t, modelsDir, "fake-pack-v1")

	// Baseline passes.
	if _, err := Verify(modelsDir, "fake-pack-v1"); err != nil {
		t.Fatal(err)
	}

	// Stray unrecorded file → loud failure.
	stray := filepath.Join(modelsDir, "fake-pack-v1", "extra.bin")
	if err := os.WriteFile(stray, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Verify(modelsDir, "fake-pack-v1"); err == nil {
		t.Fatal("unrecorded file passed verification")
	}
	os.Remove(stray)

	// Flipped byte in a member → loud failure with the member named.
	weights := filepath.Join(modelsDir, "fake-pack-v1", "model.safetensors")
	data, _ := os.ReadFile(weights)
	data[0] ^= 0xFF
	if err := os.WriteFile(weights, data, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := Verify(modelsDir, "fake-pack-v1")
	var verr *VerificationError
	if !errors.As(err, &verr) || verr.Member != "model.safetensors" {
		t.Fatalf("err = %v, want VerificationError on model.safetensors", err)
	}
}

func TestList(t *testing.T) {
	modelsDir := t.TempDir()
	installFakePack(t, modelsDir, "fake-pack-v1")
	installFakePack(t, modelsDir, "other-pack-v2")
	// Non-pack dirs (e.g. unrelated directories share ~/.futon/models) are skipped.
	if err := os.MkdirAll(filepath.Join(modelsDir, "blobs"), 0o700); err != nil {
		t.Fatal(err)
	}
	infos, err := List(modelsDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 2 {
		t.Fatalf("List = %d packs, want 2 (%+v)", len(infos), infos)
	}
	for _, info := range infos {
		if !info.OK || info.Revision != "deadbeefcafe" || info.SizeBytes == 0 {
			t.Errorf("info = %+v", info)
		}
	}
	// Missing models dir is empty, not an error.
	infos, err = List(filepath.Join(modelsDir, "nope"))
	if err != nil || infos != nil {
		t.Fatalf("List(missing) = %v, %v", infos, err)
	}
}

// ── additional unhappy paths ──────────────────────────────────────────────────

func TestInstallCorruptArchive(t *testing.T) {
	archivePath := filepath.Join(t.TempDir(), "fake-pack-v1"+ArchiveSuffix)
	if err := os.WriteFile(archivePath, []byte("this is not a zstd stream at all"), 0o600); err != nil {
		t.Fatal(err)
	}
	modelsDir := t.TempDir()
	_, err := Install(modelsDir, archivePath, false)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("corrupt archive left entries: %v", got)
	}
}

func TestResolveTruncatedDownload(t *testing.T) {
	packID := "fake-pack-v1"
	packArchive := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	archives := map[string][]byte{packID + ArchiveSuffix: packArchive}
	// The server serves only the first half of the archive; the checksums
	// manifest vouches for the whole thing.
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive[:len(packArchive)/2],
		ChecksumsAsset:         checksumsFor(archives),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()
	_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError (truncated body must fail the archive sha)", err)
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("truncated download left entries: %v", got)
	}
}

func TestResolveReleaseMissingArchiveAsset(t *testing.T) {
	// The weights release exists and has a checksums file, but this pack's archive was
	// never uploaded (partial release).
	assets := map[string][]byte{
		ChecksumsAsset: []byte("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  fake-pack-v1" + ArchiveSuffix + "\n"),
	}
	srv := fakeSource(t, assets)
	_, err := ResolveWith(Options{ModelsDir: t.TempDir(), AutoPull: true, Source: sourceFor(srv)}, "fake-pack-v1")
	var unavailable *UnavailableError
	if !errors.As(err, &unavailable) {
		t.Fatalf("err = %v, want UnavailableError", err)
	}
	if !strings.Contains(err.Error(), "fake-pack-v1"+ArchiveSuffix) {
		t.Errorf("error %q does not name the missing asset", err)
	}
}

func TestResolveUnwritableModelsDir(t *testing.T) {
	if os.Geteuid() == 0 || runtime.GOOS == "windows" {
		t.Skip("directory permission bits are not enforced here")
	}
	packID := "fake-pack-v1"
	packArchive := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	archives := map[string][]byte{packID + ArchiveSuffix: packArchive}
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         checksumsFor(archives),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()
	if err := os.Chmod(modelsDir, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(modelsDir, 0o700) })
	// A full disk and an unwritable directory take the same code path: the
	// staging write fails before anything is visible.
	_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
	if err == nil {
		t.Fatal("resolve into unwritable models dir succeeded")
	}
	if got := visibleEntries(t, modelsDir); len(got) != 0 {
		t.Fatalf("failed install left entries: %v", got)
	}
}

func TestResolveConcurrentSamePack(t *testing.T) {
	packID := "fake-pack-v1"
	packArchive := writePackArchive(t, packBytes(t, packID, fakePackFiles()))
	archives := map[string][]byte{packID + ArchiveSuffix: packArchive}
	assets := map[string][]byte{
		packID + ArchiveSuffix: packArchive,
		ChecksumsAsset:         checksumsFor(archives),
	}
	srv := fakeSource(t, assets)
	modelsDir := t.TempDir()

	const n = 8
	errs := make(chan error, n)
	for i := 0; i < n; i++ {
		go func() {
			_, err := ResolveWith(Options{ModelsDir: modelsDir, AutoPull: true, Source: sourceFor(srv)}, packID)
			errs <- err
		}()
	}
	for i := 0; i < n; i++ {
		if err := <-errs; err != nil {
			t.Errorf("concurrent resolve %d: %v", i, err)
		}
	}
	if got := visibleEntries(t, modelsDir); len(got) != 1 || got[0] != packID {
		t.Fatalf("models dir after %d concurrent resolves = %v", n, got)
	}
	if _, err := Verify(modelsDir, packID); err != nil {
		t.Fatalf("pack after concurrent installs fails verification: %v", err)
	}
	// No stray staging leftovers either.
	entries, _ := os.ReadDir(modelsDir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), ".weights-") {
			t.Errorf("leftover staging entry %s", e.Name())
		}
	}
}

// fakeSource serves assets as a static directory: GET /<asset name>.
func fakeSource(t *testing.T, assets map[string][]byte) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, ok := assets[strings.TrimPrefix(r.URL.Path, "/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func sourceFor(srv *httptest.Server) *Source {
	return &Source{BaseURL: srv.URL, HTTPClient: srv.Client()}
}
