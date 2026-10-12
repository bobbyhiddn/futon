# Design notes

## Why an in-process engine

A sidecar daemon is a hard runtime dependency with its own failure modes (stuck runners, port clashes, a download the user never asked for) and no story for hosts where nothing can be installed. futon embeds with nothing but a verified weight-pack directory on disk: no daemon, no CGO, one static binary.

## Scope

futon embeds text locally and nothing else. There is one `Embedder` interface and one engine. Hosted embedding APIs, HTTP embedding servers and daemons are out of scope: applications that want those already have clients for them. The embedder and the `packs` package contain no network code. Downloading weights lives in the separate, optional `hub` package, which only the command-line tool and applications that want a one-call "get this model" import.

## Vector provenance

The same upstream checkpoint served by different engines can produce different vectors. llama.cpp-based servers tokenize BERT text non-conformantly: against cross-checked reference vectors, simple lowercase text matches exactly, but text with accents, mixed case or punctuation drifts to as low as 0.58 cosine. futon's tokenizer matches the HuggingFace fast tokenizer id-for-id, and on real weights the engine holds >= 0.999999 cosine against the pinned reference on every corpus text.

Consequences: vectors from different engines must never be compared or mixed under one model name, and an index built by one engine must be re-embedded rather than extended by another. `ModelName()` is therefore engine-qualified (`native/<pack id>`).

## Prefix policy: none

Pack manifests carry the upstream task-prefix templates (for example `search_query: `) for provenance, but the provider never applies them. Raw text in, vector out; prefixes are a caller concern, and changing them is an index migration, not a provider default.

## Weight packs

A pack is a directory (see [packs.md](packs.md) for building and sharing them):

```
<models dir>/bge-small-en-v1.5/
  config.json        architecture + tokenizer options
  model.safetensors  fp16, canonical tensor names, strict validation on load
  vocab.txt          WordPiece vocab (line number = token id)
  manifest.json      upstream repo + revision, per-member sha256, license id
  LICENSE
```

Built by `scripts/build-weight-pack.py` from revisions pinned by commit SHA (bumping one moves the vector space, so it is a reviewed change). The script is deterministic (byte-identical re-runs) and self-verifying (bit-exact tensor round trip). It also builds any plain BERT encoder from `--repo`/`--revision`; see [packs.md](packs.md).

| pack | upstream | size |
|---|---|---|
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 | 67 MB |
| `nomic-embed-text-v1.5` | nomic-ai/nomic-embed-text-v1.5 | 274 MB |
| `snowflake-arctic-embed-m-long` | Snowflake/snowflake-arctic-embed-m-long | see manifest |

`LoadModel` verifies `manifest.json` before trusting anything: per-member sha256 and size, plus the exact tensor-name set. A pack with no manifest is refused. Unknown formats, shapes, dtypes or tensors fail loudly at load; never a zero vector, never a silently wrong architecture.

nomic is trained on 2048 positions but declares 8192; inputs are head-truncated to the trained length rather than extrapolating RoPE.

## Kernels

All dense math goes through the `Kernel` interface. Implementations: a blocked, goroutine-parallel pure-Go kernel; AVX2/FMA/F16C assembly on capable amd64 CPUs (fp16 weights converted in-register); NEON assembly on arm64 (4×4 panels, sixteen accumulators, fp16 widened in-register). `FUTON_NATIVE_PUREGO=1` forces the portable kernel.

The SIMD kernels split large GEMMs across threads by output column, so each worker streams only its slice of the weights; GEMMs under about two million multiply-adds run serially, since fan-out would cost more than it saves. When a forward's per-head attention GEMMs fall under that bound (typical embedding lengths), heads run in parallel instead, each on a serial copy of the kernel. The attention key axis is zero-padded to a multiple of eight so both attention GEMMs run whole SIMD blocks.

Softmax, GELU and SiLU use float32 kernels (`native/mathf32.go`: exp within 1 ULP, GELU within 1e-6 absolute) rather than float64 `math` calls. For CLS-pooled packs the last layer computes only the CLS row, which is bit-identical to computing every row. Forward scratch buffers are pooled per model.

Every kernel keeps one contract: thread count may decide which worker computes an output element, never the reduction order within one. Same input, same host, same kernel gives bit-identical output at any parallelism. Different kernels may round differently from each other; cross-kernel correctness is carried by the cosine parity gates.

## Batching

Each text runs as its own unpadded forward pass, and `EmbedBatch` fans texts across a bounded worker pool, longest text first. The pool shares the thread budget elastically: each GEMM runs on the budget divided by the forwards still in flight, so as the queue drains the last (longest) texts spread onto the idle cores instead of finishing on one. Without padding, padding cannot leak into attention or pooling by construction; batch and single results are bit-equal (pinned by test), since thread count never changes an output bit.

## Verification

- Tokenizer conformance against the HF fast tokenizer over a 26-text corpus (prose, code, logs, accents, emoji, CJK, RTL, edge cases), plus truncation.
- Toy-model architecture tests: two tiny random-weight packs, one per option set, against reference outputs generated from the upstream modeling code. These keep CI honest without large downloads.
- Golden parity on real weights (skipped when packs are absent), gate >= 0.999 per text.
- Determinism across thread counts.
- Strictness: malformed safetensors, tampered packs and unknown formats are all loud failures, all tested.

Fixtures come from `scripts/gen-native-embed-fixtures.py` with pinned library versions and revisions. Each golden is computed two independent ways and generation aborts unless they agree.

## Measured

Parity against the pinned reference (cross-checked two independent ways at generation time), 26 corpus texts, real fp16 packs, amd64 AVX2 kernel:

| pack | min cosine | mean cosine |
|---|---|---|
| bge-small-en-v1.5 | 1.000000 | 1.000000 |
| nomic-embed-text-v1.5 | 0.999999 | 1.000000 |
| nomic-embed-text-v1.5, matryoshka 256 | 0.999999 | n/a |
| snowflake-arctic-embed-m-long | 0.999998 | 1.000000 |

A model outside the pinned set, all-MiniLM-L6-v2 packed with `--repo`, scores a minimum of 0.999999 over 7 texts against sentence-transformers (including an over-long input clamped to its 256-token limit).

Throughput on an Apple M4 Pro (14 cores), `EmbedBatch` over the 26-text parity corpus:

| pack | one thread | all cores |
|---|---|---|
| bge-small-en-v1.5 | 54 ms/text | 12 ms/text |
| nomic-embed-text-v1.5 | 218 ms/text | 30 ms/text |
| snowflake-arctic-embed-m-long | 206 ms/text | 35 ms/text |

Numbers will vary by CPU.
