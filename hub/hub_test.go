package hub

import (
	"archive/tar"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"

	"github.com/bobbyhiddn/futon/packs"
)

// ── fixtures ──────────────────────────────────────────────────────────────────

func testArchive(t *testing.T, packID string) []byte { return testArchiveV(t, packID, "") }

// testArchiveV builds a pack whose content differs with variant.
func testArchiveV(t *testing.T, packID, variant string) []byte {
	t.Helper()
	files := map[string][]byte{
		"config.json":       []byte(`{"model_id":"` + packID + `"}`),
		"model.safetensors": bytes.Repeat([]byte{0xAB}, 4096),
		"vocab.txt":         []byte("[PAD]\n[CLS]\n[SEP]\n[UNK]\nhello\n"),
		"LICENSE":           []byte("test license " + variant + "\n"),
	}
	members := map[string]map[string]any{}
	for name, data := range files {
		sum := sha256.Sum256(data)
		members[name] = map[string]any{"sha256": hex.EncodeToString(sum[:]), "size": len(data)}
	}
	manifest, err := json.Marshal(map[string]any{
		"pack_format": 1, "model_id": packID, "members": members,
		"upstream": map[string]any{"repo": "fixture/repo", "revision": "deadbeef", "license": "mit"},
	})
	if err != nil {
		t.Fatal(err)
	}
	all := map[string][]byte{"manifest.json": manifest}
	for k, v := range files {
		all[k] = v
	}

	var buf bytes.Buffer
	zw, _ := zstd.NewWriter(&buf)
	tw := tar.NewWriter(zw)
	for name, data := range all {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data))}); err != nil {
			t.Fatal(err)
		}
		tw.Write(data)
	}
	tw.Close()
	zw.Close()
	return buf.Bytes()
}

func sum(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// host serves assets as a static directory and counts requests.
func host(t *testing.T, assets map[string][]byte) (*httptest.Server, *int) {
	t.Helper()
	n := new(int)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*n++
		data, ok := assets[strings.TrimPrefix(r.URL.Path, "/")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Write(data)
	}))
	t.Cleanup(srv.Close)
	return srv, n
}

func leftovers(t *testing.T, dir string) []string {
	t.Helper()
	entries, _ := os.ReadDir(dir)
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

// ── tests ─────────────────────────────────────────────────────────────────────

func TestPullDownloadsVerifiesAndInstalls(t *testing.T) {
	archive := testArchive(t, "fake-pack-v1")
	srv, _ := host(t, map[string][]byte{
		"fake-pack-v1.futonw": archive,
		ChecksumsName:         []byte(sum(archive) + "  fake-pack-v1.futonw\n"),
	})
	modelsDir := t.TempDir()

	var last, total int64
	dir, err := Pull(context.Background(), "fake-pack-v1", Options{
		ModelsDir: modelsDir, BaseURL: srv.URL,
		Progress: func(done, tot int64) { last, total = done, tot },
	})
	if err != nil {
		t.Fatal(err)
	}
	if dir != filepath.Join(modelsDir, "fake-pack-v1") {
		t.Errorf("dir = %q", dir)
	}
	if _, err := packs.Verify(modelsDir, "fake-pack-v1"); err != nil {
		t.Fatalf("installed pack fails verification: %v", err)
	}
	if last != int64(len(archive)) || total != int64(len(archive)) {
		t.Errorf("progress ended at %d of %d, want %d of %d", last, total, len(archive), len(archive))
	}
	if got := leftovers(t, modelsDir); len(got) != 1 || got[0] != "fake-pack-v1" {
		t.Errorf("models dir = %v, want just the pack", got)
	}
}

func TestPullRefusesAChecksumMismatchAndInstallsNothing(t *testing.T) {
	archive := testArchive(t, "fake-pack-v1")
	tampered := append([]byte{}, archive...)
	tampered[len(tampered)/2] ^= 0xFF
	srv, _ := host(t, map[string][]byte{
		"fake-pack-v1.futonw": tampered,
		ChecksumsName:         []byte(sum(archive) + "  fake-pack-v1.futonw\n"),
	})
	modelsDir := t.TempDir()
	_, err := Pull(context.Background(), "fake-pack-v1", Options{ModelsDir: modelsDir, BaseURL: srv.URL})
	var cerr *ChecksumError
	if !errors.As(err, &cerr) {
		t.Fatalf("err = %v, want *ChecksumError", err)
	}
	if got := leftovers(t, modelsDir); len(got) != 0 {
		t.Fatalf("a rejected download left %v", got)
	}
}

func TestPullTruncatedDownloadFailsTheChecksum(t *testing.T) {
	archive := testArchive(t, "fake-pack-v1")
	srv, _ := host(t, map[string][]byte{
		"fake-pack-v1.futonw": archive[:len(archive)/2],
		ChecksumsName:         []byte(sum(archive) + "  fake-pack-v1.futonw\n"),
	})
	modelsDir := t.TempDir()
	_, err := Pull(context.Background(), "fake-pack-v1", Options{ModelsDir: modelsDir, BaseURL: srv.URL})
	var cerr *ChecksumError
	if !errors.As(err, &cerr) {
		t.Fatalf("err = %v, want *ChecksumError", err)
	}
	if got := leftovers(t, modelsDir); len(got) != 0 {
		t.Fatalf("a truncated download left %v", got)
	}
}

func TestPullUnlistedPackIsTypedAndFetchesNoArchive(t *testing.T) {
	archive := testArchive(t, "other")
	srv, n := host(t, map[string][]byte{
		"other.futonw": archive,
		ChecksumsName:  []byte(sum(archive) + "  other.futonw\n"),
	})
	_, err := Pull(context.Background(), "fake-pack-v1", Options{ModelsDir: t.TempDir(), BaseURL: srv.URL})
	var nh *NotHostedError
	if !errors.As(err, &nh) || nh.ID != "fake-pack-v1" {
		t.Fatalf("err = %v, want *NotHostedError", err)
	}
	if *n != 1 {
		t.Errorf("made %d requests, want 1 (only the checksums file)", *n)
	}
}

func TestPullHostErrorsAreReported(t *testing.T) {
	srv, _ := host(t, map[string][]byte{}) // everything 404s
	_, err := Pull(context.Background(), "fake-pack-v1", Options{ModelsDir: t.TempDir(), BaseURL: srv.URL})
	if err == nil || !strings.Contains(err.Error(), "404") {
		t.Fatalf("err = %v, want a 404", err)
	}
	srv.Close() // now unreachable
	if _, err := Pull(context.Background(), "fake-pack-v1", Options{ModelsDir: t.TempDir(), BaseURL: srv.URL}); err == nil {
		t.Fatal("pulling from a dead host succeeded")
	}
}

func TestPullHonoursContextCancellation(t *testing.T) {
	archive := testArchive(t, "fake-pack-v1")
	srv, _ := host(t, map[string][]byte{
		"fake-pack-v1.futonw": archive,
		ChecksumsName:         []byte(sum(archive) + "  fake-pack-v1.futonw\n"),
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	modelsDir := t.TempDir()
	if _, err := Pull(ctx, "fake-pack-v1", Options{ModelsDir: modelsDir, BaseURL: srv.URL}); err == nil {
		t.Fatal("a cancelled pull succeeded")
	}
	if got := leftovers(t, modelsDir); len(got) != 0 {
		t.Fatalf("a cancelled pull left %v", got)
	}
}

func TestPullIsIdempotentAndNeverClobbersADifferentPack(t *testing.T) {
	archive := testArchive(t, "fake-pack-v1")
	srv, _ := host(t, map[string][]byte{
		"fake-pack-v1.futonw": archive,
		ChecksumsName:         []byte(sum(archive) + "  fake-pack-v1.futonw\n"),
	})
	modelsDir := t.TempDir()
	opts := Options{ModelsDir: modelsDir, BaseURL: srv.URL}
	if _, err := Pull(context.Background(), "fake-pack-v1", opts); err != nil {
		t.Fatal(err)
	}
	if _, err := Pull(context.Background(), "fake-pack-v1", opts); err != nil {
		t.Fatalf("pulling the same pack again should succeed: %v", err)
	}

	// A host serving a different pack under the same id must not replace what is installed.
	other := testArchiveV(t, "fake-pack-v1", "different")
	srv2, _ := host(t, map[string][]byte{
		"fake-pack-v1.futonw": other,
		ChecksumsName:         []byte(sum(other) + "  fake-pack-v1.futonw\n"),
	})
	before, _ := os.ReadFile(filepath.Join(modelsDir, "fake-pack-v1", "manifest.json"))
	_, err := Pull(context.Background(), "fake-pack-v1", Options{ModelsDir: modelsDir, BaseURL: srv2.URL})
	var already *packs.AlreadyInstalledError
	if !errors.As(err, &already) {
		t.Fatalf("err = %v, want *packs.AlreadyInstalledError", err)
	}
	after, _ := os.ReadFile(filepath.Join(modelsDir, "fake-pack-v1", "manifest.json"))
	if !bytes.Equal(before, after) {
		t.Fatal("the installed pack was replaced")
	}
}

func TestPullRejectsBadInput(t *testing.T) {
	for _, id := range []string{"", "../x", "a/b", ".hidden"} {
		if _, err := Pull(context.Background(), id, Options{ModelsDir: t.TempDir(), BaseURL: "http://127.0.0.1:1"}); err == nil {
			t.Errorf("pack id %q accepted", id)
		}
	}
	if _, err := Pull(context.Background(), "x", Options{BaseURL: "http://127.0.0.1:1"}); err == nil {
		t.Error("missing models dir accepted")
	}
	if _, err := Pull(context.Background(), "x", Options{ModelsDir: t.TempDir(), BaseURL: "file:///etc"}); err == nil {
		t.Error("non-http base URL accepted")
	}
}

func TestAvailableListsOnlyWellFormedArchiveEntries(t *testing.T) {
	good := strings.Repeat("a", 64)
	srv, _ := host(t, map[string][]byte{
		ChecksumsName: []byte(fmt.Sprintf("%s  one.futonw\n%s *two.futonw\nnothex%s  three.futonw\n%s  notes.txt\nshort  four.futonw\n", good, good, strings.Repeat("a", 58), good)),
	})
	got, err := Available(context.Background(), Options{BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].ID != "one" || got[1].ID != "two" {
		t.Fatalf("Available = %+v, want one and two only", got)
	}
}
