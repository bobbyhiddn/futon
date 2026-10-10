# futon and goformer

[goformer](https://github.com/MichaelAyles/goformer) is the closest project: a pure-Go BERT-family inference library that loads safetensors weights. This page records how the two differ, measured, so you can choose. It is accurate for goformer v0.1.0 (March 2026); check its README for anything newer.

## Where goformer is better for you

- **Simpler start.** `goformer.Load(dir)` reads a Hugging Face model directory as it is: `config.json`, `tokenizer.json` and a safetensors file. futon needs a one-time conversion into a weight pack with a Python script.
- **Smaller.** About 600 lines of Go, easy to read in an afternoon.
- **Tokenizer from `tokenizer.json`.** goformer parses the model's own tokenizer file; futon reads `vocab.txt` and the tokenizer options its pack builder copies from the model card.

## Where futon differs

| | goformer v0.1.0 | futon |
|---|---|---|
| Matrix kernel | tiled scalar Go | AVX2/FMA (amd64), NEON (arm64), parallel pure-Go fallback; fp16 weights converted in-register |
| Pooling | mean, fixed | `cls` or `mean`, per model |
| Architectures | BERT (absolute positions, GELU) | BERT, plus nomic-bert (RoPE, SwiGLU) with matryoshka truncation |
| Weights | HF directory as is | verified packs (sha256 per file, exact tensor set), atomic install |
| Over-long input | model's declared position count | per-pack clamp (honours `sentence_bert_config.json` and the trained context) |
| Determinism | not stated | bit-identical at any thread count, pinned by test |
| Extras | none | command line, pack builder for any BERT encoder, reference checker, CI on Linux/macOS/Windows incl. arm64 |

## Measured

Same machine (16-thread Windows, amd64), same five texts (prose, a question, accents, code, a long paragraph), `bge-small-en-v1.5` at the same pinned revision, one text per call:

| | ms per text | min cosine vs sentence-transformers `bge-small-en-v1.5` |
|---|---|---|
| goformer v0.1.0 | 858 | 0.9207 |
| futon | 52 | 1.0000 |

The speed gap is the kernels. The cosine gap is pooling: `bge-small-en-v1.5` is a CLS-pooled model, and goformer always mean-pools. Against a mean-pooled bge reference, goformer scores 0.999999, so it is faithful to its own reference; but for CLS-pooled models (bge, snowflake-arctic-embed) its vectors are not the model's published embeddings. For mean-pooled models (all-MiniLM, e5, gte) the pooling matches.

Numbers are from one machine and one small corpus; run your own before deciding.
