<p align="center"><img src="docs/logo.png" alt="futon" width="260"></p>

# futon

Text in, vectors out. futon runs embedding models inside your Go program: pure Go, no CGO, no sidecar, and no network at embed time.

It is a library, not a server and not an LLM runtime. It does one job, turning text into vectors with BERT-family encoders, so applications can add vector search, clustering or memory without shipping or managing a separate embedding service.

- **In process.** Weights load from a local directory into your process. Nothing to install, start or keep alive.
- **Pure Go.** No CGO anywhere in the dependency tree (CI enforces it), so cross-compiling is `GOOS=... go build`. AVX2/FMA kernels on amd64, NEON on arm64, a portable fallback everywhere else.
- **Faithful.** The tokenizer matches the Hugging Face reference id for id, and vectors match the reference implementation at cosine >= 0.9999 (typically >= 0.999999) on a corpus of prose, code, accents, CJK, emoji and edge cases. Output is bit-deterministic at any thread count.
- **Bring your own model.** Any plain BERT encoder works once packed (see [docs/packs.md](docs/packs.md)); three popular models are tested out of the box.
- **Careful with weights.** Packs are sha256-verified at install and again at load and installed atomically. The embedder never touches the network; downloading is a separate, optional step (`hub` package, `futon packs pull`).

## Try it in a minute

You need Go 1.23+. No Python.

```
go install github.com/bobbyhiddn/futon/cmd/futon@latest
futon packs pull bge-small-en-v1.5     # 61 MB, checksum-verified, installed to ~/.futon/models
futon embed "hello world"              # prints a 384-number vector
```

Then in your own program: `go get github.com/bobbyhiddn/futon` and see below.

## Use

```go
import "github.com/bobbyhiddn/futon"

e, err := futon.New(futon.Options{Model: "bge-small-en-v1.5"})
if err != nil {
    // *packs.NotInstalledError if the pack is not on disk
}
defer e.Close()

vec, err := e.Embed(ctx, "the quick brown fox")
vecs, err := e.EmbedBatch(ctx, texts)

e.Dimensions() // 384
e.ModelName()  // "native/bge-small-en-v1.5": store this next to every vector
```

`CosineSimilarity`, `Serialize` and `Deserialize` are included for the common ranking and storage steps. futon deliberately has no index or database: use whatever store you already have.

A model is a *weight pack*, a directory under `~/.futon/models` (override with `Options.ModelsDir` or `FUTON_HOME`). Get one by:

- `futon packs pull <id>` for the [hosted models](MODELS.md) (MIT and Apache-2.0 models published by others, redistributed with their licenses and provenance). In Go, `hub.Pull(ctx, id, hub.Options{ModelsDir: dir})` does the same with progress reporting.
- building any plain BERT model yourself with `scripts/build-weight-pack.py`; see [docs/packs.md](docs/packs.md), which also covers checking a pack against the reference and sharing packs between machines.

The embedder itself never downloads anything: an application that ships its own weights carries no network code.

## Models

| pack | size | dims | vs reference (min cosine, 26 texts) |
|---|---|---|---|
| `bge-small-en-v1.5` | 67 MB | 384 | 1.000000 |
| `nomic-embed-text-v1.5` | 274 MB | 768 (matryoshka to 64) | 0.999999 |
| `snowflake-arctic-embed-m-long` | 274 MB | 768 | 0.999998 |
| `all-MiniLM-L6-v2` (built with `--repo`) | 45 MB | 384 | 0.999999 (7 texts) |

Supported: plain BERT encoders (WordPiece vocab, absolute positions, GELU) with CLS or mean pooling, and the nomic-bert variant (RoPE, SwiGLU). Not supported: GPU, training, decoder/LLM models, SentencePiece or BPE tokenizers (so no multilingual-e5 or other XLM-R models), and anything that is not an encoder.

Speed depends on your CPU. On an Apple M4 Pro (14 cores), embedding the mixed-content parity corpus with `EmbedBatch` takes about 12 ms per text for bge-small and 30-35 ms for nomic and arctic, or 54 ms and 205-220 ms on a single thread; run `go test ./native -bench Native` with a pack installed to measure your own hardware.

## Command line

```
go install github.com/bobbyhiddn/futon/cmd/futon@latest

futon embed -model bge-small-en-v1.5 "some text" "more text"   # one JSON vector per line
futon packs available                    # models hosted for download
futon packs pull bge-small-en-v1.5       # download, verify, install
futon packs list
futon packs verify bge-small-en-v1.5
futon packs install my-model.futonw      # from an archive made by the build script
futon packs load ./models/my-model       # from a pack directory
```

## Vector spaces

Vectors from different engines are different spaces, even for the same checkpoint: llama.cpp-based servers tokenize BERT text non-conformantly and drift from the reference. Never compare or mix vectors under one model name. `ModelName()` is engine-qualified for this reason, so a store can detect a mismatch and re-embed. Changing a pack's pinned revision also moves the space.

## Related projects

futon is not the first Go library to run embedding models locally. Pick the one that matches your constraints:

- [goformer](https://github.com/MichaelAyles/goformer): pure-Go BERT inference that loads a Hugging Face model directory as is, with no conversion step, so it is quicker to try. See [docs/comparison.md](docs/comparison.md) for how it differs (kernels, pooling, architectures, packs).
- [hugot](https://github.com/knights-analytics/hugot): Hugging Face pipelines for Go, ONNX models, with a pure-Go backend (GoMLX) or ONNX Runtime/XLA backends that need native libraries. Broader than futon: more pipelines and architectures.
- [cybertron](https://github.com/nlpodyssey/cybertron): pure-Go transformers on the spaGO framework, with its own converted model format.
- [kelindar/search](https://github.com/kelindar/search): GGUF models through llama.cpp shared libraries, plus a vector index.
- [fastembed-go](https://github.com/Anush008/fastembed-go): ONNX Runtime bindings.

## Layout

| path | what |
|---|---|
| `/` | `Embedder`, `Options`, `New`, vector helpers |
| `native/` | the engine: tokenizer, safetensors, kernels, model |
| `packs/` | pack install, verify, list (offline) |
| `hub/` | optional: download hosted packs with checksum verification |
| `cmd/futon` | command-line tool |
| `scripts/` | pack builder, pack checker, fixture generator |
| `MODELS.md` | hosted models: upstream, revision, license, checksum |
| `docs/` | [pack guide](docs/packs.md), [design notes](docs/design.md), [comparison](docs/comparison.md) |

## Testing

`go test ./...` runs toy-model architecture, tokenizer, kernel and pack tests with no downloads. CI runs it on Linux (amd64, arm64), macOS (arm64) and Windows with Go 1.23 and stable, with `CGO_ENABLED=0`, with the portable kernel, and under the race detector, plus real-weights parity against bge-small on Linux and macOS (which exercises the AVX2 and NEON kernels on real hardware).

To run parity on every model locally:

```
python scripts/build-weight-pack.py --models-dir ./models
FUTON_NATIVE_TEST_MODELS=./models go test ./native -run Parity -v
```

Requires Go 1.23 or newer.

## Acknowledgements

Several of the engine's performance techniques (the float32 exp kernel, column-split GEMM threading, the 16-accumulator NEON panel, CLS-only final layer, head-parallel attention) are adapted from [aikit](https://github.com/townsendmerino/aikit) by Francis Townsend-Merino, under the MIT License. See [NOTICE](NOTICE).

## License

MIT for the code. Weight packs are converted copies of models published by others, under their own licenses (recorded in each pack's `LICENSE` and `manifest.json`, summarized in [MODELS.md](MODELS.md)); check them before redistributing packs.
