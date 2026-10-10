# Weight packs

futon runs *weight packs*: a model's tensors, vocabulary and a manifest, converted once into a directory futon can load and verify. You can download the [hosted models](../MODELS.md) with `futon packs pull`, or build a pack from any plain BERT model yourself. The embedder never downloads anything; only the optional `hub` package and the command-line tool do.

## Pack format

```
<models dir>/<pack id>/
  config.json        architecture and tokenizer options
  model.safetensors  fp16, canonical tensor names
  vocab.txt          WordPiece vocabulary (line number = token id)
  manifest.json      upstream repo + revision, per-file sha256, tensor names, license
  LICENSE            license id and provenance
```

The pack id is the directory name and the value of `Options.Model`. The default models directory is `~/.futon/models` (override with `Options.ModelsDir`, `FUTON_HOME`, or `-models-dir` on the command line).

At load futon checks every file against `manifest.json` (sha256 and size) and the exact tensor-name set. A missing manifest, a changed byte or an unknown tensor is a loud error, never a silently different model.

## Download a hosted model

```
futon packs available                   # what is hosted
futon packs pull bge-small-en-v1.5      # download, verify the sha256, install
```

`pull` fetches `<id>.futonw` and `weights-checksums.txt` from the release listed in [MODELS.md](../MODELS.md), checks the archive against the manifest before unpacking anything, then installs it atomically. `-source URL` points it at any other host with the same layout (a mirror, an internal server).

From Go: `hub.Pull(ctx, id, hub.Options{ModelsDir: dir, Progress: fn})`. The `hub` package is the only code in futon that uses the network.

## Build the built-in models yourself

```
pip install huggingface_hub safetensors numpy zstandard
python scripts/build-weight-pack.py --model bge       # bge-small-en-v1.5    (67 MB)
python scripts/build-weight-pack.py --model nomic     # nomic-embed-text-v1.5 (274 MB)
python scripts/build-weight-pack.py --model arctic    # snowflake-arctic-embed-m-long (274 MB)
python scripts/build-weight-pack.py                   # all three
```

Revisions are pinned by commit sha inside the script, and the output is byte-reproducible. Add `--models-dir DIR` to build somewhere other than `~/.futon/models`.

## Bring your own model

Any plain BERT encoder can be packed: absolute position embeddings, exact GELU, and a WordPiece `vocab.txt`. That covers most small retrieval models (bge-base/large, all-MiniLM-L6/L12, e5 (English), gte, mxbai-embed and similar). It does not cover models that use SentencePiece or BPE tokenizers (multilingual-e5, other XLM-R based models), decoder models, or architectures other than BERT and nomic-bert.

```
python scripts/build-weight-pack.py \
    --repo sentence-transformers/all-MiniLM-L6-v2 \
    --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
    --pooling mean
```

- `--revision` must be the full 40-character commit sha, never a branch name. The pin defines the vector space: a different revision is a different model, so re-embed anything stored with the old one.
- `--pooling` is how the model turns token vectors into one vector: `cls` or `mean`. It is part of how the model was trained; read the model card. Getting it wrong produces plausible but wrong vectors, which is why the next step exists.
- `--pack-id NAME` sets the pack id (default: the repo name, lowercased).
- `--no-normalize` is for the rare model that does not L2-normalize its output.
- `--license ID` records a license id; by default it is read from the model card.
- If the model has a `sentence_bert_config.json`, the script clamps inputs to its `max_seq_length`, which is usually shorter than BERT's 512 positions. Without it, long inputs would diverge from the reference.

### Check the pack against the reference

```
go build -o futon ./cmd/futon
pip install torch transformers sentence-transformers
python scripts/check-pack.py \
    --repo sentence-transformers/all-MiniLM-L6-v2 \
    --revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
    --pack-id all-minilm-l6-v2 --futon ./futon
```

It embeds prose, code, accents, CJK, an empty string and an over-long input with both implementations and prints the cosine for each. A correct pack scores >= 0.9999 on every text; the models above score >= 0.999999. If a text scores low, the usual causes are the wrong `--pooling`, a missing `--no-normalize`, or a tokenizer option the model card sets that the script did not carry over.

Packs built this way are not part of futon's own test corpus, so run the check yourself and keep the result with the pack.

## Move packs between machines

```
python scripts/build-weight-pack.py --model bge --archive --archive-dir dist
# copy dist/bge-small-en-v1.5.futonw anywhere, then on the other machine:
futon packs install bge-small-en-v1.5.futonw
futon packs verify bge-small-en-v1.5
```

Archives are deterministic tar.zst files. `futon packs install` unpacks into a hidden staging directory, verifies every member against the manifest, then renames atomically; a bad archive leaves nothing behind. The same operations are in Go as `packs.Install`, `packs.LoadDir`, `packs.Verify` and `packs.List`.

## Updating a model

There is no in-place update. To move to a newer upstream revision, build a new pack (give it a new `--pack-id`, for example `bge-small-en-v1.5-r2`), re-embed your data, and remove the old pack when nothing references it. Vectors from two revisions must not be mixed.

## Adding an architecture

The engine implements BERT (absolute positions, GELU) and nomic-bert (RoPE, SwiGLU, no biases). Another encoder architecture means changes in `native/` (model, config) and a tensor-name mapping in `scripts/build-weight-pack.py`, plus golden fixtures from `scripts/gen-native-embed-fixtures.py` so it is held to the same parity bar. The toy-model tests in `native/` show how architectures are pinned without large downloads.

## Licenses

Each pack records its upstream license in `LICENSE` and `manifest.json`. The weights stay under their original terms: check them before you redistribute a pack or ship one inside a product.
