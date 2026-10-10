//go:build windows

package packs

// syncDir is a no-op on Windows: directories cannot be opened for fsync, and
// NTFS journals the rename itself.
func syncDir(string) error { return nil }
