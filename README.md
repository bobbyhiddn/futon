<p align="center"><img src="docs/logo.png" alt="futon" width="260"></p>

# futon

Text in, vectors out. futon is the embedding half of a retrieval system and nothing more.

- **No sidecar.** A pure-Go (zero CGO) engine runs BERT-family encoders from local weight packs, in process. AVX2/FMA kernels on amd64, NEON on arm64, a portable fallback everywhere else.
- **Faithful.** Tokenization matches the HuggingFace reference exactly. On real weights the engine holds >= 0.999999 cosine against the pinned reference on every corpus text, and its output is bit-deterministic at any thread count.
- **Swappable.** One `Provider` interface. Hosted and OpenAI-compatible HTTP backends (`openai`, `gemini`, `bedrock`) sit behind the same interface, and a `null` provider lets callers fall back to lexical search.
- **Careful with weights.** Packs are sha256-verified at install and again at load, installed atomically, and never downloaded unless the host application says so.

Supported packs:

| pack | size | dims | parity vs reference (26 texts, min cosine) | speed, 16 threads, amd64 AVX2 |
|---|---|---|---|---|
| `bge-small-en-v1.5` | 67 MB | 384 | 1.000000 | ~45 ms/text, ~1,560 tokens/s |
| `nomic-embed-text-v1.5` | 274 MB | 768 (matryoshka down to 64) | 0.999999 | ~165-250 ms/text, ~300-450 tokens/s |
| `snowflake-arctic-embed-m-long` | 274 MB | 768 | 0.999998 | similar to nomic |

Speed is one Windows machine, batch of mixed prose/code/unicode texts; yours will differ. Run `go test ./native -bench Native` with packs installed to measure your own.

## Use

```go
p, err := futon.New(futon.Config{
    Provider: "native",
    Model:    "bge-small-en-v1.5",
    ModelsDir: modelsDir, // one subdirectory per pack id
})
if err != nil { /* typed: *packs.NotInstalledError, *packs.VerificationError, ... */ }
defer p.Close()

vec, err := p.Embed(ctx, "the quick brown fox")
vecs, err := p.EmbedBatch(ctx, texts)
p.ModelName() // "native/bge-small-en-v1.5": engine-qualified, store it with every vector
```

`Provider: "auto"` uses the native engine when a pack is present and returns the null provider otherwise.

## Weight packs

Packs are directories (`config.json`, `model.safetensors`, `vocab.txt`, `manifest.json`, `LICENSE`) built by `scripts/build-weight-pack.py` from pinned upstream revisions, and distributed as `.futonw` archives (deterministic tar.zst) next to a `weights-checksums.txt`.

- `packs.Install` sideloads an archive with no network.
- `packs.LoadDir` installs a pack directory.
- `packs.ResolveWith` fetches from a `packs.Source` (any static host) only when `AutoPull` is explicitly true.

## Vector spaces

Vectors from different engines are different spaces, even for the same checkpoint: llama.cpp-based servers tokenize BERT text non-conformantly. Never compare or mix vectors under one model name. `ModelName()` is engine-qualified for this reason, so a store can detect mismatches and re-embed.

## Layout

| path | what |
|---|---|
| `/` | `Provider`, `Config`, factory, HTTP backends, vector helpers |
| `native/` | the in-process engine: tokenizer, safetensors, kernels |
| `packs/` | pack install, verify, fetch |
| `scripts/` | pack builder and parity-fixture generator |
| `docs/design-native.md` | design notes and measurements |

## Testing

`go test ./...` runs toy-model architecture, tokenizer, kernel and pack tests with no downloads. Real-weights parity and benchmarks skip until packs exist:

```
python scripts/build-weight-pack.py --models-dir ./models --model bge
FUTON_NATIVE_TEST_MODELS=./models go test ./native -run Parity -v
```

With all three packs built, the suite checks every model against the reference on all 26 corpus texts, plus matryoshka truncation, the trained-context clamp and batch-equals-single bit equality.

## Docs

- [`docs/design.md`](docs/design.md): why in-process, vector provenance, pack format, kernels, verification.

## Requirements

Go 1.23 or newer. No CGO. Building packs needs Python 3.10+ (`scripts/build-weight-pack.py` lists its pinned dependencies); regenerating the test fixtures additionally needs torch (see `scripts/gen-native-embed-fixtures.py`).

## License

MIT. Weight packs are built from upstream models under their own licenses (recorded in each pack's `LICENSE` and `manifest.json`); check them before redistributing packs.
