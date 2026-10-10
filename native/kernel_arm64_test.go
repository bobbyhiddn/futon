//go:build arm64

package native

import "testing"

// TestNEONSelectedOnArm64 pins that arm64 hosts actually RUN the NEON kernel.
// Without this, a feature-detection regression would make newArchKernel
// return nil and every Arch* kernel test would silently t.Skip — CI (the
// macos-latest Apple-silicon runners are this package's real-arm64 gate, and
// darwin-arm64 is a production platform) would go green while the
// build quietly ran the scalar fallback. ASIMD is baseline ARMv8-A, so on
// any real arm64 host this must select NEON.
func TestNEONSelectedOnArm64(t *testing.T) {
	if !neonAvailable() {
		t.Fatal("neonAvailable() = false on an arm64 host — feature detection is broken (ASIMD is baseline ARMv8-A)")
	}
	k := newArchKernel(1)
	if k == nil {
		t.Fatal("newArchKernel returned nil on arm64 — NEON kernel not selected")
	}
	if _, ok := k.(*neonKernel); !ok {
		t.Fatalf("arch kernel is %T, want *neonKernel", k)
	}
	if _, ok := k.(q8Kernel); !ok {
		t.Fatal("NEON kernel does not implement the experimental q8Kernel seam")
	}
}
