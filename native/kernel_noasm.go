//go:build !amd64 && !arm64

package native

// newArchKernel returns nil on architectures without an assembly kernel;
// the portable goKernel carries correctness everywhere.
func newArchKernel(threads int) Kernel { return nil }
