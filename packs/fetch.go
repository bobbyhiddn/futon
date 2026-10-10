package packs

import (
	"crypto/sha256"
	"crypto/tls"
	"encoding/hex"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// Source says where weight packs can be downloaded from: a directory URL that
// holds <pack-id>+ArchiveSuffix archives and a ChecksumsAsset file in
// sha256sum format. Any static host works (release assets, object storage, a
// LAN mirror).
type Source struct {
	// BaseURL is the directory URL, with or without a trailing slash.
	BaseURL string
	// Token, when set, is sent as a bearer token.
	Token string
	// HTTPClient overrides the default client (tests, proxies).
	HTTPClient *http.Client
}

func (s *Source) client() *http.Client {
	if s.HTTPClient != nil {
		return s.HTTPClient
	}
	// HTTP/1.1 only: large downloads through some CDN redirect chains hit
	// HTTP/2 protocol errors in Go's client.
	return &http.Client{
		Timeout: 15 * time.Minute,
		Transport: &http.Transport{
			ForceAttemptHTTP2:   false,
			TLSNextProto:        map[string]func(string, *tls.Conn) http.RoundTripper{},
			DialContext:         (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			TLSHandshakeTimeout: 10 * time.Second,
		},
	}
}

func (s *Source) assetURL(name string) (string, error) {
	u, err := url.Parse(s.BaseURL)
	if err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" {
		return "", fmt.Errorf("packs: source base URL %q is not an http(s) URL", s.BaseURL)
	}
	u.Path = strings.TrimRight(u.Path, "/") + "/" + name
	return u.String(), nil
}

func (s *Source) get(rawURL string) (io.ReadCloser, error) {
	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/octet-stream")
	if s.Token != "" {
		req.Header.Set("Authorization", "Bearer "+s.Token)
	}
	resp, err := s.client().Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, rawURL)
	}
	return resp.Body, nil
}

func (s *Source) getAll(rawURL string, maxBytes int64) ([]byte, error) {
	body, err := s.get(rawURL)
	if err != nil {
		return nil, err
	}
	defer body.Close()
	data, err := io.ReadAll(io.LimitReader(body, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("download exceeds %d-byte limit", maxBytes)
	}
	return data, nil
}

// getToFile streams an asset to w, hashing as it goes, and returns the sha256.
func (s *Source) getToFile(rawURL string, w io.Writer, maxBytes int64) (string, error) {
	body, err := s.get(rawURL)
	if err != nil {
		return "", err
	}
	defer body.Close()
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(w, h), io.LimitReader(body, maxBytes+1))
	if err != nil {
		return "", err
	}
	if n > maxBytes {
		return "", fmt.Errorf("download exceeds %d-byte limit", maxBytes)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// pull downloads <packID>+ArchiveSuffix and the checksums file, verifies the
// archive against the checksum, extracts, verifies the pack manifest, and
// atomically installs into dir.
func pull(opts Options, packID, dir string) error {
	src := opts.Source
	archiveName := packID + ArchiveSuffix
	checksumsURL, err := src.assetURL(ChecksumsAsset)
	if err != nil {
		return &UnavailableError{PackID: packID, Err: err}
	}
	archiveURL, err := src.assetURL(archiveName)
	if err != nil {
		return &UnavailableError{PackID: packID, Err: err}
	}

	checksums, err := src.getAll(checksumsURL, maxChecksumsBytes)
	if err != nil {
		return &UnavailableError{PackID: packID, Err: fmt.Errorf("downloading %s: %w", ChecksumsAsset, err)}
	}
	wantSHA, err := checksumFor(checksums, archiveName)
	if err != nil {
		// The host serves the checksums file but does not vouch for this
		// archive: a verification failure, not an availability one.
		return &VerificationError{PackID: packID, Reason: err.Error()}
	}

	if err := os.MkdirAll(opts.ModelsDir, 0o700); err != nil {
		return fmt.Errorf("packs: create models dir: %w", err)
	}
	tmpArchive, err := os.CreateTemp(opts.ModelsDir, ".weights-download-*")
	if err != nil {
		return fmt.Errorf("packs: create download temp: %w", err)
	}
	tmpPath := tmpArchive.Name()
	defer os.Remove(tmpPath)

	gotSHA, err := src.getToFile(archiveURL, tmpArchive, maxArchiveBytes)
	if closeErr := tmpArchive.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return &UnavailableError{PackID: packID, Err: fmt.Errorf("downloading %s: %w", archiveName, err)}
	}
	if gotSHA != wantSHA {
		return &VerificationError{PackID: packID, Reason: fmt.Sprintf("archive sha256 %s does not match %s entry %s", gotSHA, ChecksumsAsset, wantSHA)}
	}

	_, tmpDir, err := extractAndVerify(opts.ModelsDir, tmpPath, packID)
	if err != nil {
		return err
	}
	return commit(opts.ModelsDir, tmpDir, dir, packID)
}
