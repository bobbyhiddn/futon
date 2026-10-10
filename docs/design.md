# Design notes

## Why an in-process engine

A sidecar daemon is a hard runtime dependency with its own failure modes (stuck runners, port clashes, a download the user never asked for) and no story for hosts where nothing can be installed. futon embeds with nothing but a verified weight-pack directory on disk: no daemon, no CGO, one static binary.

## Providers

- `native` (default): in-process inference from a weight pack. `ModelPath` points at a pack directory; otherwise `ModelsDir` joined with the pack id mapped from `Model`. The provider itself never downloads.
- `auto`: native when a usable pack exists, otherwise the null provider, so callers keep their lexical search path. Never probes the network.
- `openai`, `gemini`, `bedrock`: hosted backends. `openai` with `BaseURL` also covers any OpenAI-compatible local server.
- `null`: returns `(nil, nil)`; callers fall back to full-text search.

## Vector provenance

The same upstream checkpoint served by different engines can produce different vectors. llama.cpp-based servers tokenize BERT text non-conformantly: against cross-checked reference vectors, simple lowercase text matches exactly, but text with accents, mixed case or punctuation drifts to as low as 0.58 cosine. futon's tokenizer matches the HuggingFace fast tokenizer id-for-id, and on real weights the engine holds >= 0.999999 cosine against the pinned reference on every corpus text.

Consequences: vectors from different engines must never be compared or mixed under one model name, and an index built by one engine must be re-embedded rather than extended by another. `ModelName()` is therefore engine-qualified (`native/<pack id>`).

## Prefix policy: none

Pack manifests carry the upstream task-prefix templates (for example `search_query: `) for provenance, but the provider never applies them. Raw text in, vector out; prefixes are a caller concern, and changing them is an index migration, not a provider default.

## Weight packs

A pack is a directory:

```
<models dir>/bge-small-en-v1.5/
  config.json        architecture + tokenizer options
  model.safetensors  fp16, canonical tensor names, strict validation on load
  vocab.txt          WordPiece vocab (line number = token id)
  manifest.json      upstream repo + revision, per-member sha256, license id
  LICENSE
```

Built by `scripts/build-weight-pack.py` from revisions pinned by commit SHA (bumping one moves the vector space, so it is a reviewed change). The script is deterministic (byte-identical re-runs) and self-verifying (bit-exact tensor round trip).

| pack | upstream | size |
|---|---|---|
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 | 67 MB |
| `nomic-embed-text-v1.5` | nomic-ai/nomic-embed-text-v1.5 | 274 MB |
| `snowflake-arctic-embed-m-long` | Snowflake/snowflake-arctic-embed-m-long | see manifest |

`LoadModel` verifies `manifest.json` before trusting anything: per-member sha256 and size, plus the exact tensor-name set. A pack with no manifest is refused. Unknown formats, shapes, dtypes or tensors fail loudly at load; never a zero vector, never a silently wrong architecture.

nomic is trained on 2048 positions but declares 8192; inputs are head-truncated to the trained length rather than extrapolating RoPE.

## Kernels

All dense math goes through the `Kernel` interface. Implementations: a blocked, goroutine-parallel pure-Go kernel; AVX2/FMA/F16C assembly on capable amd64 CPUs (fp16 weights converted in-register); NEON assembly on arm64. `FUTON_NATIVE_PUREGO=1` forces the portable kernel.

Every kernel keeps one contract: thread count may decide which worker computes an output element, never the reduction order within one. Same input, same host, same kernel gives bit-identical output at any parallelism. Different kernels may round differently from each other; cross-kernel correctness is carried by the cosine parity gates.

## Batching

Each text runs as its own unpadded forward pass, and `EmbedBatch` fans texts across a bounded worker pool. Without padding, padding cannot leak into attention or pooling by construction; batch and single results are bit-equal (pinned by test).

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

Throughput on a 16-thread Windows machine, mixed-corpus batches: bge-small 210 ms/text on one thread and about 45 ms/text on all cores (about 1,560 tokens/s); nomic 590 ms/text on one thread and 165-250 ms/text on all cores (the all-core figure is noisy on a shared machine). Numbers will vary by CPU.
