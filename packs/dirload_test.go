package packs

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// writePackDir materializes a member map as an on-disk pack directory — the
// source shape LoadDir consumes.
func writePackDir(t *testing.T, members map[string][]byte) string {
	t.Helper()
	dir := t.TempDir()
	for name, data := range members {
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestLoadDirInstallsAVerifiedPack(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "fixture-pack", fakePackFiles()))

	packID, err := LoadDir(modelsDir, src, false)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if packID != "fixture-pack" {
		t.Fatalf("packID = %q", packID)
	}
	info, err := Verify(modelsDir, packID)
	if err != nil {
		t.Fatalf("post-load Verify: %v", err)
	}
	if !info.OK || info.Revision != "deadbeefcafe" {
		t.Fatalf("verify info = %+v", info)
	}
	// The source dir must be untouched (it was copied, not moved).
	if _, err := os.Stat(filepath.Join(src, "manifest.json")); err != nil {
		t.Fatalf("source dir disturbed: %v", err)
	}
}

func TestLoadDirIgnoresStrayFilesInSource(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "fixture-pack", fakePackFiles()))
	if err := os.WriteFile(filepath.Join(src, "notes-from-build.txt"), []byte("scratch"), 0o644); err != nil {
		t.Fatal(err)
	}
	packID, err := LoadDir(modelsDir, src, false)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(modelsDir, packID, "notes-from-build.txt")); !os.IsNotExist(err) {
		t.Fatal("stray file rode into the installed pack")
	}
}

func TestLoadDirRefusesTamperedMember(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "fixture-pack", fakePackFiles()))
	if err := os.WriteFile(filepath.Join(src, "vocab.txt"), []byte("tampered\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadDir(modelsDir, src, false)
	var verr *VerificationError
	if !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
	// Nothing may be visible in models_dir — no partial pack, no stage litter.
	entries, _ := os.ReadDir(modelsDir)
	for _, e := range entries {
		t.Fatalf("models dir not empty after refused load: %s", e.Name())
	}
}

func TestLoadDirRefusesMissingMember(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "fixture-pack", fakePackFiles()))
	if err := os.Remove(filepath.Join(src, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	var verr *VerificationError
	if _, err := LoadDir(modelsDir, src, false); !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
}

func TestLoadDirNoClobberContract(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "fixture-pack", fakePackFiles()))
	if _, err := LoadDir(modelsDir, src, false); err != nil {
		t.Fatal(err)
	}

	// Identical pack: no-op, no error.
	if packID, err := LoadDir(modelsDir, src, false); err != nil || packID != "fixture-pack" {
		t.Fatalf("identical reload: %v %q", err, packID)
	}

	// Differing pack: refused without force, replaced with it.
	files := fakePackFiles()
	files["vocab.txt"] = []byte("[PAD]\n[CLS]\n[SEP]\n[UNK]\nchanged\n")
	src2 := writePackDir(t, packBytes(t, "fixture-pack", files))
	var already *AlreadyInstalledError
	if _, err := LoadDir(modelsDir, src2, false); !errors.As(err, &already) {
		t.Fatalf("differing reload: %v, want AlreadyInstalledError", err)
	}
	if _, err := LoadDir(modelsDir, src2, true); err != nil {
		t.Fatalf("forced replace: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(modelsDir, "fixture-pack", "vocab.txt"))
	if err != nil || string(got) != string(files["vocab.txt"]) {
		t.Fatalf("replace did not land: %v %q", err, got)
	}
}

func TestLoadDirRefusesItsOwnInstalledLocation(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "fixture-pack", fakePackFiles()))
	if _, err := LoadDir(modelsDir, src, false); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDir(modelsDir, filepath.Join(modelsDir, "fixture-pack"), true); err == nil {
		t.Fatal("loading a pack from its own installed location must refuse")
	}
}

func TestLoadDirRefusesInvalidModelID(t *testing.T) {
	modelsDir := t.TempDir()
	src := writePackDir(t, packBytes(t, "../escape", fakePackFiles()))
	var verr *VerificationError
	if _, err := LoadDir(modelsDir, src, false); !errors.As(err, &verr) {
		t.Fatalf("err = %v, want VerificationError", err)
	}
}
