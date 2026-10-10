# Hosted models

These weight packs are published as assets of the [`weights-v1` release](https://github.com/bobbyhiddn/futon/releases/tag/weights-v1) so you can install them with one command:

```
futon packs pull bge-small-en-v1.5
```

They are converted copies of models published by others, redistributed under the licenses their authors chose. futon's own MIT license covers the code, not these weights.

| pack id | upstream model | revision | license | dims | pooling | archive | sha256 of archive |
|---|---|---|---|---|---|---|---|
| `bge-small-en-v1.5` | [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | MIT | 384 | cls | 61 MB | `4cf0bdc0a8c7c5ee4b9e2b1f98ef1216e21a48238e0184906b0feed4f06fd824` |
| `all-minilm-l6-v2` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Apache-2.0 | 384 | mean | 42 MB | `0a48af360f1145e1ec460867fda2560e4b71edf65c8b4a78ed77bd1cc6839b92` |
| `nomic-embed-text-v1.5` | [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | `e9b6763023c676ca8431644204f50c2b100d9aab` | Apache-2.0 | 768 (matryoshka to 64) | mean | 252 MB | `08afe7a56ca6f487f50810117029e02ea02f37d1ae5bb9b8f22e2e6a14f69101` |
| `snowflake-arctic-embed-m-long` | [Snowflake/snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) | `92d97331f1f4b6a366c1f161354b9f3390cc219f` | Apache-2.0 | 768 | cls | 252 MB | `e2b025ee452946d78447ad45ecfad9594120c563957d6696d03941ae911cb5f7` |

The archives are deterministic: rebuilding a pack from the pinned revision with `scripts/build-weight-pack.py` produces the same bytes, and the checksums above are what `futon packs pull` verifies before installing anything.

## What was changed

For each model the build script downloads the upstream `model.safetensors` and `vocab.txt` at the revision above, casts the weights to 16-bit floats, renames and packs tensors into futon's layout (for BERT models the separate query, key and value matrices are concatenated), and records provenance in a manifest. Nothing is retrained, fine-tuned or quantized beyond the float16 cast. The vectors match the upstream reference implementations (cosine 0.999998 or better on the test corpus).

## Licenses and attribution

Each pack contains a `LICENSE` file naming the license, the upstream repository and the revision, and a `manifest.json` with the same provenance. The upstream repositories declare their licenses in the model card metadata rather than shipping a separate license file, so the pack points at the SPDX text:

- MIT: https://spdx.org/licenses/MIT.html
- Apache-2.0: https://spdx.org/licenses/Apache-2.0.html

Credit for the models belongs to their authors: BAAI (bge), the Sentence-Transformers project (all-MiniLM-L6-v2), Nomic AI (nomic-embed-text) and Snowflake (arctic-embed). Check the upstream model cards for the full terms, training-data notes and citation requests before using a model in a product.

## Adding a model here

A model qualifies if its license permits redistribution of converted weights and it can be packed by `scripts/build-weight-pack.py` (see [docs/packs.md](docs/packs.md)). Open an issue with the model, its license, and the output of `scripts/check-pack.py`.
