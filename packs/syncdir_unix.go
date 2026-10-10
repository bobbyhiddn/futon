//go:build !windows

package packs

import "os"

// syncDir flushes a directory's entries to disk after a rename, so an atomic
// install survives power loss.
func syncDir(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}
