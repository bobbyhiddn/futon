// Package hub downloads weight packs from a static host and installs them.
//
// It is the only part of futon that touches the network, and it is optional:
// the embedder and the packs package never import it, so an application that
// ships its own weights carries no network code. The command-line tool and
// applications that want a one-call "get this model" use it.
//
// A host is a directory URL holding <pack id>.futonw archives and a
// weights-checksums.txt in sha256sum format. DefaultBaseURL is the release
// that hosts the models listed in MODELS.md; any other static host with the
// same layout works (Options.BaseURL).
package hub

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/bobbyhiddn/futon/packs"
)

// DefaultBaseURL is the release asset directory holding the hosted models.
const DefaultBaseURL = "https://github.com/bobbyhiddn/futon/releases/download/weights-v1"

// ChecksumsName is the sha256 manifest next to the archives.
const ChecksumsName = "weights-checksums.txt"

const (
	maxChecksumsBytes = int64(1 << 20) // 1 MiB
	maxArchiveBytes   = int64(2 << 30) // 2 GiB
)

// Options configures a download.
type Options struct {
	// ModelsDir is where the pack is installed. Required.
	ModelsDir string
	// BaseURL is the host directory. Empty means DefaultBaseURL.
	BaseURL string
	// HTTPClient overrides the default client (tests, proxies).
	HTTPClient *http.Client
	// Progress, when set, is called as the archive downloads. total is -1 when the
	// host does not say how large the file is.
	Progress func(done, total int64)
}

// Model is one downloadable pack.
type Model struct {
	ID     string
	SHA256 string
}

// ChecksumError reports an archive whose hash does not match the host's manifest.
// Nothing is installed.
type ChecksumError struct {
	ID        string
	Want, Got string
}

func (e *ChecksumError) Error() string {
	return fmt.Sprintf("hub: %s archive sha256 %s does not match the manifest entry %s; nothing was installed", e.ID, e.Got, e.Want)
}

// NotHostedError reports a pack the host's manifest does not list.
type NotHostedError struct{ ID string }

func (e *NotHostedError) Error() string {
	return fmt.Sprintf("hub: %q is not hosted at this source (see Available for what is)", e.ID)
}

func (o Options) base() string {
	if o.BaseURL != "" {
		return o.BaseURL
	}
	return DefaultBaseURL
}

func (o Options) client() *http.Client {
	if o.HTTPClient != nil {
		return o.HTTPClient
	}
	// HTTP/1.1 only: large downloads through some CDN redirect chains hit HTTP/2
	// protocol errors in Go's client. No overall timeout: the context bounds it.
	return &http.Client{
		Transport: &http.Transport{
			Proxy:               http.ProxyFromEnvironment,
			ForceAttemptHTTP2:   false,
			TLSNextProto:        map[string]func(string, *tls.Conn) http.RoundTripper{},
			DialContext:         (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			TLSHandshakeTimeout: 10 * time.Second,
		},
	}
}

func (o Options) assetURL(name string) (string, error) {
	u, err := url.Parse(o.base())
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return "", fmt.Errorf("hub: base URL %q is not an http(s) URL", o.base())
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/" + name
	return u.String(), nil
}

func (o Options) get(ctx context.Context, rawURL string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/octet-stream")
	resp, err := o.client().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, rawURL)
	}
	return resp, nil
}

// Available lists the packs the host offers, from its checksums manifest.
func Available(ctx context.Context, opts Options) ([]Model, error) {
	data, err := fetchChecksums(ctx, opts)
	if err != nil {
		return nil, err
	}
	return parseChecksums(data), nil
}

func fetchChecksums(ctx context.Context, opts Options) ([]byte, error) {
	u, err := opts.assetURL(ChecksumsName)
	if err != nil {
		return nil, err
	}
	resp, err := opts.get(ctx, u)
	if err != nil {
		return nil, fmt.Errorf("hub: reading %s: %w", ChecksumsName, err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxChecksumsBytes+1))
	if err != nil {
		return nil, fmt.Errorf("hub: reading %s: %w", ChecksumsName, err)
	}
	if int64(len(data)) > maxChecksumsBytes {
		return nil, fmt.Errorf("hub: %s exceeds %d bytes", ChecksumsName, maxChecksumsBytes)
	}
	return data, nil
}

// parseChecksums reads sha256sum-format lines, keeping entries for archives.
func parseChecksums(data []byte) []Model {
	var out []Model
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 2 {
			continue
		}
		name := filepath.Base(strings.TrimPrefix(fields[1], "*"))
		id, ok := strings.CutSuffix(name, packs.ArchiveSuffix)
		sum := strings.ToLower(fields[0])
		if !ok || len(sum) != 64 {
			continue
		}
		if _, err := hex.DecodeString(sum); err != nil {
			continue
		}
		out = append(out, Model{ID: id, SHA256: sum})
	}
	return out
}

// Pull downloads the pack id, verifies the archive against the host's manifest, and
// installs it atomically into opts.ModelsDir. It returns the installed pack directory.
// Pulling a pack that is already installed unchanged succeeds; a different pack under the
// same id is never replaced and is reported as *packs.AlreadyInstalledError.
func Pull(ctx context.Context, id string, opts Options) (string, error) {
	if opts.ModelsDir == "" {
		return "", errors.New("hub: models dir is required")
	}
	if err := packs.ValidatePackID(id); err != nil {
		return "", err
	}
	data, err := fetchChecksums(ctx, opts)
	if err != nil {
		return "", err
	}
	var want string
	for _, m := range parseChecksums(data) {
		if m.ID == id {
			want = m.SHA256
			break
		}
	}
	if want == "" {
		return "", &NotHostedError{ID: id}
	}

	archiveURL, err := opts.assetURL(id + packs.ArchiveSuffix)
	if err != nil {
		return "", err
	}
	resp, err := opts.get(ctx, archiveURL)
	if err != nil {
		return "", fmt.Errorf("hub: downloading %s: %w", id, err)
	}
	defer resp.Body.Close()

	// The download lives outside the models dir; Install stages and verifies it again.
	tmpDir, err := os.MkdirTemp("", "futon-pull-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmpDir)
	tmpPath := filepath.Join(tmpDir, id+packs.ArchiveSuffix)
	f, err := os.OpenFile(tmpPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", err
	}

	h := sha256.New()
	total := resp.ContentLength
	body := io.LimitReader(resp.Body, maxArchiveBytes+1)
	var done int64
	buf := make([]byte, 256<<10)
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				return "", werr
			}
			h.Write(buf[:n])
			done += int64(n)
			if opts.Progress != nil {
				opts.Progress(done, total)
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			return "", fmt.Errorf("hub: downloading %s: %w", id, rerr)
		}
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	if done > maxArchiveBytes {
		return "", fmt.Errorf("hub: %s exceeds %d bytes", id, maxArchiveBytes)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return "", &ChecksumError{ID: id, Want: want, Got: got}
	}

	if _, err := packs.Install(opts.ModelsDir, tmpPath, false); err != nil {
		return "", err
	}
	return filepath.Join(opts.ModelsDir, id), nil
}
