#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch==2.4.1",
#   "transformers==4.44.2",
#   "sentence-transformers==3.0.1",
#   "einops==0.8.0",
#   "huggingface_hub==0.34.4",
#   "numpy<2",
# ]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu" }]
# ///
"""Generate the test fixtures for package native.

Everything is pinned (model revisions, library versions above, RNG seeds), so
re-running regenerates byte-comparable fixtures.

Outputs (under native/testdata/):
  tokenizer_fixtures.json  — HF fast-tokenizer ids for the corpus, per model,
                             plus truncation pinning at max_len=16
  parity/<model>.json      — golden embeddings for the corpus. CROSS-CHECKED:
                             computed via sentence-transformers AND via raw
                             AutoModel + manual pooling; both paths must agree
                             (cosine >= 0.9999 per text) or generation aborts.
                             The stored golden is the raw-transformers path.
  toy/<pack>/...           — two tiny random-weight packs (2 layers, 32 dim)
                             exercising absolute+GELU+CLS and RoPE+SwiGLU+mean,
                             F32 safetensors (the real packs cover F16)
  toy/expected.json        — reference outputs for the toy packs: forward
                             passes over explicit id sequences AND end-to-end
                             text cases through the HF tokenizer

Requires network for the pinned model downloads. CI never runs this; it runs
on a dev box and the fixtures are committed.
"""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTDATA = REPO_ROOT / "native" / "testdata"

NOMIC_REPO = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
BGE_REPO = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# snowflake-arctic-embed-m-long ships its own pinned copy of the nomic-bert
# modeling code IN-REPO, so the model revision pins the code too.
ARCTIC_REPO = "Snowflake/snowflake-arctic-embed-m-long"
ARCTIC_REVISION = "92d97331f1f4b6a366c1f161354b9f3390cc219f"
# Pinned revision of the shared nomic-bert modeling code (used to build the
# RoPE+SwiGLU toy reference).
NOMIC_BERT_REPO = "nomic-ai/nomic-bert-2048"
NOMIC_BERT_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"

SEED = 7
CROSS_CHECK_MIN = 0.9999

# ── Corpus ───────────────────────────────────────────────────────────────────
# Diverse texts + the edge cases the verification bar demands. Kept in one
# place so tokenizer fixtures, parity fixtures, and Go benchmarks all share it.
CORPUS: list[tuple[str, str]] = [
    ("prose_short", "The quick brown fox jumps over the lazy dog."),
    ("prose_question", "How does the recall pipeline decide between FTS5 and vector search?"),
    ("prose_long", "A note-taking assistant maintains long-lived conversational memory "
     "while worker sessions execute delegated tasks. "
     "Memory is captured as typed records, embedded into a vector index, and recalled "
     "into constructed context when a new turn arrives. The quality of that recall "
     "depends directly on the fidelity of the embedding engine, which is why parity "
     "between inference stacks is treated as a release gate rather than a nicety."),
    ("code_go", "func (p *Provider) Embed(ctx context.Context, text string) ([]float32, error) {\n"
     "\tids := p.tok.Encode(text, p.model.Config().MaxSeqLen)\n"
     "\treturn p.model.EncodeTokens(ctx, ids, p.dims)\n}"),
    ("code_python", "def cosine(a, b):\n    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))"),
    ("code_js", "const handler = async (req) => { const { id } = await req.json(); return db.get(id) ?? null; };"),
    ("code_sql", "SELECT chunk_id, snippet(fts, 0, '[', ']', '…', 12) FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 20;"),
    ("code_yaml", "embedding:\n  provider: auto\n  model: nomic-embed-text\n  models_dir: ~/.futon/models\n  timeout: 5m"),
    ("log_line", "2026-08-16T19:33:12Z WARN embed auto: native pack missing, falling back to lexical search model=nomic-embed-text-v1.5 dims=768"),
    ("markdown", "## Recall\n\n- **FTS5** for exact identifiers\n- *vector* search for semantics\n- hybrid rerank when both fire"),
    ("error_msg", "safetensors: tensor \"layers.7.attn.qkv.weight\" shape [2304 768] × F16 needs 3538944 bytes, offsets span 3538940"),
    ("numbers", "In Q3 the index grew from 1,204,547 to 9,882,113 vectors (≈8.2x) at p99 latency 43ms."),
    ("unicode_accents", "Élodie Müller façade naïve coöperate señor Đặng Trần Hưng Đạo café crème brûlée"),
    ("unicode_emoji", "shipping it 🚀🔥 the tests are green ✅ but the build is on fire 🧯"),
    ("cjk", "自然言語処理は面白い。 机器学习模型需要大量数据。 자연어 처리는 재미있다."),
    ("mixed_rtl", "The word سلام means peace and שלום means the same."),
    ("single_token", "hello"),
    ("single_char", "x"),
    ("repeated_token", " ".join(["buffalo"] * 100)),
    ("punct_run", "(((([[{{}}]]))))->=>!=<=>=&&||::…—``\"\"''@#$%^&*~~~///\\\\\\"),
    ("unk_leading", "☃ snowman leads this sentence"),
    ("empty", ""),
    ("whitespace_only", " \t\n  \t "),
    ("exactly_max_bge", " ".join(["cat"] * 510)),
    ("over_max_bge", " ".join(["cat"] * 600)),
    ("tabs_and_controls", "col1\tcol2\tcol3\nrow\x07with\x00controls"),
    # Special-token literals in input text. HF's fast tokenizer matches the
    # added tokens ([CLS], [SEP], [PAD], [UNK], [MASK]) as literal substrings
    # of the INPUT and emits their ids — callers index source code, which
    # contains exactly such strings in tokenizer tests. These two cases pin
    # that behavior (and that non-special [unusedN] vocab entries do NOT get
    # literal-matched).
    ("code_special_literals", "seeds := []string{\"[CLS] injected [SEP]\", \"[MASK]ed\", \"pad token: [PAD]\"}\n"
     "if ids[0] != clsID { t.Fatalf(\"want [CLS], got [UNK]\") }"),
    ("special_adversarial", "[unused0] [CLS]x[SEP]"),
]

# trust_remote_code / code_revision: nomic's modeling code lives in a
# SEPARATE repo (nomic-bert-2048), so the AutoModel path pins code_revision
# explicitly; arctic carries the modeling code in its own repo, where the
# model revision pins the code as well.
MODELS = {
    "nomic-embed-text-v1.5": {
        "repo": NOMIC_REPO,
        "revision": NOMIC_REVISION,
        "max_seq_len": 8192,
        "pooling": "mean",
        "matryoshka_dim": 256,
        "trust_remote_code": True,
        "code_revision": NOMIC_BERT_REVISION,
    },
    "bge-small-en-v1.5": {
        "repo": BGE_REPO,
        "revision": BGE_REVISION,
        "max_seq_len": 512,
        "pooling": "cls",
        "matryoshka_dim": None,
        "trust_remote_code": False,
        "code_revision": None,
    },
    "snowflake-arctic-embed-m-long": {
        "repo": ARCTIC_REPO,
        "revision": ARCTIC_REVISION,
        "max_seq_len": 8192,
        "pooling": "cls",
        "matryoshka_dim": None,
        "trust_remote_code": True,
        "code_revision": ARCTIC_REVISION,
        # Arctic's in-repo nomic-bert loader ignores `revision` (and defaults
        # to pytorch_model.bin, which the repo doesn't ship) when it fetches
        # weights from the hub itself. Loading from a local snapshot of the
        # PINNED revision takes its direct-safetensors branch instead —
        # fully pinned, no loader quirks.
        "load_via_snapshot": True,
    },
}


def model_source(meta: dict) -> tuple[str, dict]:
    """Return (path_or_repo, revision_kwargs) for model loading: the hub repo
    plus pinned-revision kwargs, or a local snapshot of the pinned revision
    for repos whose custom loader mishandles hub fetches (see MODELS)."""
    if meta.get("load_via_snapshot"):
        from huggingface_hub import snapshot_download

        return snapshot_download(meta["repo"], revision=meta["revision"]), {}
    return meta["repo"], {"revision": meta["revision"]}


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a, b) / (na * nb))


def round_floats(vec) -> list[float]:
    # 7 significant digits keeps fixture files compact; the induced cosine
    # error is ~1e-7, far inside the 0.999 gate's margin.
    return [float(f"{v:.7g}") for v in vec]


# ── Deterministic safetensors writer (F32) — mirror of build-weight-pack.py ──
def write_safetensors(tensors: dict[str, np.ndarray], out: Path, dtype: str) -> None:
    np_dtype, tag, size = {"F32": (np.float32, "F32", 4), "F16": (np.float16, "F16", 2)}[dtype]
    names = sorted(tensors)
    header: dict[str, dict] = {}
    offset = 0
    blobs = []
    for name in names:
        arr = np.ascontiguousarray(tensors[name].astype(np_dtype))
        blob = arr.astype("<" + {"F32": "f4", "F16": "f2"}[dtype], copy=False).tobytes()
        header[name] = {"dtype": tag, "shape": list(arr.shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
        blobs.append(blob)
    hjson = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    hjson += b" " * ((8 - (8 + len(hjson)) % 8) % 8)
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for blob in blobs:
            f.write(blob)


# ── Tokenizer fixtures ───────────────────────────────────────────────────────
def gen_tokenizer_fixtures() -> None:
    from transformers import AutoTokenizer

    out: dict = {"generator": "scripts/gen-native-embed-fixtures.py", "models": {}}
    for model_id, meta in MODELS.items():
        tok = AutoTokenizer.from_pretrained(meta["repo"], revision=meta["revision"], use_fast=True)
        cases = []
        for name, text in CORPUS:
            ids = tok(text, truncation=True, max_length=meta["max_seq_len"])["input_ids"]
            ids16 = tok(text, truncation=True, max_length=16)["input_ids"]
            cases.append({"name": name, "text": text, "ids": ids, "ids_max16": ids16})
        out["models"][model_id] = {"max_seq_len": meta["max_seq_len"], "cases": cases}
    path = TESTDATA / "tokenizer_fixtures.json"
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")


# ── Parity fixtures (two independent reference paths, cross-checked) ─────────
# PARITY_EXCLUDE: the two special-token-literal texts are TOKENIZER fixtures
# only, never parity goldens. Measured 2026-08-17: the two reference paths
# genuinely disagree on them for bge (inter-reference cosine 0.947 on
# code_special_literals) because sentence-transformers instantiates the
# tokenizer with the repo's sentence_bert_config args and then does NOT
# literal-match special tokens in input text, while the raw AutoTokenizer
# path does. futon's canonical semantics is the AutoTokenizer path's
# literal matching — pinned directly by the tokenizer-conformance fixtures
# (which DO include these texts) — so parity generation keeps the committed
# 26-text baseline where both references agree, rather than weakening the
# cross-check or silently overriding sentence-transformers' tokenizer.
PARITY_EXCLUDE = {"code_special_literals", "special_adversarial"}
def reference_transformers(model_id: str, meta: dict, texts: list[str]) -> np.ndarray:
    """Path B: raw AutoModel forward + our own pooling/normalization in numpy."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(meta["repo"], revision=meta["revision"], use_fast=True)
    src, rev_kwargs = model_source(meta)
    model = AutoModel.from_pretrained(
        src,
        trust_remote_code=meta["trust_remote_code"],
        # Pin the remote modeling code too — trust_remote_code would
        # otherwise track its repo's main branch (see MODELS). For local
        # snapshots the code is already the pinned revision's.
        code_revision=(meta["code_revision"] if rev_kwargs else None),
        **rev_kwargs,
    )
    model.eval()
    vecs = []
    for text in texts:
        enc = tok(text, truncation=True, max_length=meta["max_seq_len"], return_tensors="pt")
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state[0].numpy()  # [seq, h] — no padding: batch of one
        if meta["pooling"] == "mean":
            pooled = hidden.mean(axis=0)
        else:
            pooled = hidden[0]
        pooled = pooled / np.linalg.norm(pooled)
        vecs.append(pooled)
    return np.stack(vecs)


def reference_sentence_transformers(model_id: str, meta: dict, texts: list[str]) -> np.ndarray:
    """Path A: sentence-transformers pipeline (the ecosystem-canonical path)."""
    from sentence_transformers import SentenceTransformer

    src, rev_kwargs = model_source(meta)
    st = SentenceTransformer(
        src,
        trust_remote_code=meta["trust_remote_code"],
        device="cpu",
        **rev_kwargs,
    )
    vecs = st.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)
    return vecs.astype(np.float64)


def gen_parity_fixtures(only_model: str = "all") -> None:
    (TESTDATA / "parity").mkdir(parents=True, exist_ok=True)
    for model_id, meta in MODELS.items():
        if only_model != "all" and model_id != only_model:
            continue
        names = [n for n, _ in CORPUS if n not in PARITY_EXCLUDE]
        texts = [t for n, t in CORPUS if n not in PARITY_EXCLUDE]
        a = reference_sentence_transformers(model_id, meta, texts)
        b = reference_transformers(model_id, meta, texts)
        worst = 1.0
        for i, name in enumerate(names):
            c = cos(a[i], b[i])
            worst = min(worst, c)
            if c < CROSS_CHECK_MIN:
                raise SystemExit(
                    f"{model_id}: reference paths disagree on {name!r}: cosine {c:.6f} < {CROSS_CHECK_MIN}"
                )
        fixture = {
            "model_id": model_id,
            "revision": meta["revision"],
            "generator": "scripts/gen-native-embed-fixtures.py",
            "reference": "transformers-AutoModel (cross-checked vs sentence-transformers, "
                         f"min inter-reference cosine {worst:.6f})",
            "cases": [
                {"name": names[i], "text": texts[i], "vector": round_floats(b[i])}
                for i in range(len(texts))
            ],
        }
        if meta["matryoshka_dim"]:
            dim = meta["matryoshka_dim"]
            # The published nomic v1.5 resizing recipe: parameter-free
            # layer_norm on the UN-normalized pooled vector -> slice -> L2.
            # Recompute pooled (pre-normalize) from path B for exactness.
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(meta["repo"], revision=meta["revision"], use_fast=True)
            model = AutoModel.from_pretrained(meta["repo"], revision=meta["revision"], trust_remote_code=True,
                                              code_revision=NOMIC_BERT_REVISION)
            model.eval()
            mvecs = []
            for text in texts:
                enc = tok(text, truncation=True, max_length=meta["max_seq_len"], return_tensors="pt")
                with torch.no_grad():
                    hidden = model(**enc).last_hidden_state[0]
                pooled = hidden.mean(dim=0, keepdim=True)
                pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
                pooled = pooled[:, :dim]
                pooled = F.normalize(pooled, p=2, dim=1)[0].numpy()
                mvecs.append(round_floats(pooled))
            fixture["matryoshka"] = {"dim": dim, "vectors": mvecs}
        path = TESTDATA / "parity" / f"{model_id}.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path}  (inter-reference min cosine {worst:.6f})")


def write_toy_manifest(pack_dir: Path, model_id: str, tensor_names: list[str]) -> None:
    """Toy packs carry the same integrity manifest real packs do: the Go
    loader refuses any pack without a verifiable manifest.json (LOW-2)."""
    import hashlib

    def sha256_file(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    members = {}
    for member in ("config.json", "model.safetensors", "vocab.txt"):
        p = pack_dir / member
        members[member] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    manifest = {
        "pack_format": 1,
        "model_id": model_id,
        "created_by": "scripts/gen-native-embed-fixtures.py",
        "members": members,
        "tensor_names": sorted(tensor_names),
    }
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


# ── Toy packs ────────────────────────────────────────────────────────────────
TOY_VOCAB = (
    ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    + ["the", "cat", "sat", "on", "mat", "hello", "world", "snow", "man",
       "un", "runs", "affable", "##aff", "##able", "##s", "##ing", "##man",
       "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
       "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
       ".", ",", "!", "?", "(", ")", "{", "}", ";", ":", "-"]
)
TOY_TEXTS = [
    "the cat sat on the mat.",
    "hello unaffable world!",
    "snowman ☃ runs",
    "",
]
# Explicit id sequences for the pure-forward test (no tokenizer involved):
# CLS=2, SEP=3. Lengths chosen to cross the GEMM micro-kernel's 4-row
# boundary and reach the toy max_seq_len of 64.
TOY_ID_SEQS = [
    [2, 3],
    [2, 5, 6, 7, 8, 5, 9, 44, 3],
    [2] + [(i % 50) + 5 for i in range(31)] + [3],
    [2] + [(i * 7 % 50) + 5 for i in range(62)] + [3],
]

TOY_COMMON = {
    "hidden": 32, "layers": 2, "heads": 4, "inter": 64,
    # NomciBertGatedMLP rounds hidden_features up to a multiple of 256, so
    # the RoPE+SwiGLU toy uses 256 to keep config and weights consistent.
    "inter_swiglu": 256,
    "vocab": 64, "types": 2, "max_seq": 64, "eps": 1e-12,
}


def toy_pack_config(model_id: str, position: str, activation: str, pooling: str, matryoshka: bool) -> dict:
    c = TOY_COMMON
    inter = c["inter_swiglu"] if activation == "swiglu" else c["inter"]
    cfg = {
        "pack_format": 1,
        "model_id": model_id,
        "hidden_size": c["hidden"],
        "num_layers": c["layers"],
        "num_heads": c["heads"],
        "intermediate_size": inter,
        "vocab_size": c["vocab"],
        "type_vocab_size": c["types"],
        "max_seq_len": c["max_seq"],
        "layer_norm_eps": c["eps"],
        "position_encoding": position,
        "activation": activation,
        "pooling": pooling,
        "normalize": True,
        "tokenizer": {
            "do_lower_case": True,
            "tokenize_chinese_chars": True,
            "cls_token": "[CLS]", "sep_token": "[SEP]",
            "unk_token": "[UNK]", "pad_token": "[PAD]",
        },
    }
    if position == "rope":
        cfg["rope_theta"] = 1000.0
    if matryoshka:
        cfg["matryoshka"] = {"layernorm": True, "dims": [32, 16]}
    return cfg


def build_toy_bert() -> tuple[dict, torch.nn.Module, "object"]:
    """Absolute+GELU+CLS toy on the REAL transformers BertModel."""
    from transformers import BertConfig, BertModel, BertTokenizer

    c = TOY_COMMON
    torch.manual_seed(SEED)
    config = BertConfig(
        vocab_size=c["vocab"], hidden_size=c["hidden"], num_hidden_layers=c["layers"],
        num_attention_heads=c["heads"], intermediate_size=c["inter"],
        max_position_embeddings=c["max_seq"], type_vocab_size=c["types"],
        hidden_act="gelu", layer_norm_eps=c["eps"],
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )
    model = BertModel(config, add_pooling_layer=False)
    model.eval()

    sd = {k: v.numpy() for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {
        "embeddings.word": sd["embeddings.word_embeddings.weight"],
        "embeddings.position": sd["embeddings.position_embeddings.weight"],
        "embeddings.token_type": sd["embeddings.token_type_embeddings.weight"],
        "embeddings.norm.weight": sd["embeddings.LayerNorm.weight"],
        "embeddings.norm.bias": sd["embeddings.LayerNorm.bias"],
    }
    for i in range(c["layers"]):
        s, d = f"encoder.layer.{i}", f"layers.{i}"
        for kind in ("weight", "bias"):
            out[f"{d}.attn.qkv.{kind}"] = np.concatenate(
                [sd[f"{s}.attention.self.query.{kind}"],
                 sd[f"{s}.attention.self.key.{kind}"],
                 sd[f"{s}.attention.self.value.{kind}"]], axis=0)
            out[f"{d}.attn.out.{kind}"] = sd[f"{s}.attention.output.dense.{kind}"]
            out[f"{d}.norm1.{kind}"] = sd[f"{s}.attention.output.LayerNorm.{kind}"]
            out[f"{d}.mlp.fc1.{kind}"] = sd[f"{s}.intermediate.dense.{kind}"]
            out[f"{d}.mlp.fc2.{kind}"] = sd[f"{s}.output.dense.{kind}"]
            out[f"{d}.norm2.{kind}"] = sd[f"{s}.output.LayerNorm.{kind}"]
    return out, model, BertTokenizer


def build_toy_nomic() -> tuple[dict, torch.nn.Module]:
    """RoPE+SwiGLU+mean toy on the PINNED upstream nomic-bert modeling code —
    the reference is genuinely independent of our Go implementation."""
    from huggingface_hub import hf_hub_download

    cfg_py = hf_hub_download(NOMIC_BERT_REPO, "configuration_hf_nomic_bert.py", revision=NOMIC_BERT_REVISION)
    mdl_py = hf_hub_download(NOMIC_BERT_REPO, "modeling_hf_nomic_bert.py", revision=NOMIC_BERT_REVISION)

    # The modeling file uses a relative import, so it must live in a package.
    import shutil
    import tempfile

    pkg_root = Path(tempfile.mkdtemp(prefix="nomic-bert-ref-"))
    pkg_dir = pkg_root / "nomic_bert_ref"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    shutil.copy(cfg_py, pkg_dir / "configuration_hf_nomic_bert.py")
    shutil.copy(mdl_py, pkg_dir / "modeling_hf_nomic_bert.py")
    sys.path.insert(0, str(pkg_root))
    conf_mod = importlib.import_module("nomic_bert_ref.configuration_hf_nomic_bert")
    model_mod = importlib.import_module("nomic_bert_ref.modeling_hf_nomic_bert")

    c = TOY_COMMON
    torch.manual_seed(SEED + 1)
    config = conf_mod.NomicBertConfig(
        n_embd=c["hidden"], n_head=c["heads"], n_layer=c["layers"], n_inner=c["inter_swiglu"],
        vocab_size=c["vocab"], type_vocab_size=c["types"],
        n_positions=c["max_seq"], max_position_embeddings=c["max_seq"],
        activation_function="swiglu",
        qkv_proj_bias=False, mlp_fc1_bias=False, mlp_fc2_bias=False,
        rotary_emb_fraction=1.0, rotary_emb_base=1000, rotary_emb_interleaved=False,
        rotary_emb_scale_base=None, rotary_scaling_factor=None,
        prenorm=False, causal=False, use_flash_attn=True, fused_bias_fc=False,
        fused_dropout_add_ln=False, pad_vocab_size_multiple=1,
        attn_pdrop=0.0, resid_pdrop=0.0, embd_pdrop=0.0,
        layer_norm_epsilon=c["eps"], pad_token_id=0, use_rms_norm=False,
        dense_seq_output=False, use_xentropy=False,
    )
    model = model_mod.NomicBertModel(config, add_pooling_layer=False)
    model.eval()

    sd = {k: v.numpy() for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {
        "embeddings.word": sd["embeddings.word_embeddings.weight"],
        "embeddings.token_type": sd["embeddings.token_type_embeddings.weight"],
        "embeddings.norm.weight": sd["emb_ln.weight"],
        "embeddings.norm.bias": sd["emb_ln.bias"],
    }
    for i in range(c["layers"]):
        s, d = f"encoder.layers.{i}", f"layers.{i}"
        out[f"{d}.attn.qkv.weight"] = sd[f"{s}.attn.Wqkv.weight"]
        out[f"{d}.attn.out.weight"] = sd[f"{s}.attn.out_proj.weight"]
        out[f"{d}.norm1.weight"] = sd[f"{s}.norm1.weight"]
        out[f"{d}.norm1.bias"] = sd[f"{s}.norm1.bias"]
        out[f"{d}.mlp.fc1.weight"] = sd[f"{s}.mlp.fc11.weight"]
        out[f"{d}.mlp.gate.weight"] = sd[f"{s}.mlp.fc12.weight"]
        out[f"{d}.mlp.fc2.weight"] = sd[f"{s}.mlp.fc2.weight"]
        out[f"{d}.norm2.weight"] = sd[f"{s}.norm2.weight"]
        out[f"{d}.norm2.bias"] = sd[f"{s}.norm2.bias"]
    return out, model


def toy_forward_expected(model, ids: list[int], pooling: str, matryoshka_dim: int | None = None) -> list[float]:
    input_ids = torch.tensor([ids])
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        hidden = model(input_ids=input_ids, attention_mask=attn).last_hidden_state[0]
    pooled = hidden.mean(dim=0, keepdim=True) if pooling == "mean" else hidden[0:1]
    if matryoshka_dim:
        pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
        pooled = pooled[:, :matryoshka_dim]
    pooled = F.normalize(pooled, p=2, dim=1)[0]
    return round_floats(pooled.numpy())


def gen_toys() -> None:
    toy_dir = TESTDATA / "toy"
    expected: dict = {"generator": "scripts/gen-native-embed-fixtures.py", "packs": {}}

    # ── absolute + gelu + cls ──
    pack = toy_dir / "toy-absolute-gelu"
    pack.mkdir(parents=True, exist_ok=True)
    tensors, bert, BertTokenizer = build_toy_bert()
    write_safetensors(tensors, pack / "model.safetensors", "F32")
    cfg = toy_pack_config("toy-absolute-gelu", "absolute", "gelu", "cls", matryoshka=False)
    (pack / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (pack / "vocab.txt").write_text("\n".join(TOY_VOCAB) + "\n", encoding="utf-8", newline="\n")
    write_toy_manifest(pack, "toy-absolute-gelu", list(tensors))

    vocab_file = pack / "vocab.txt"
    tok = BertTokenizer(str(vocab_file), do_lower_case=True)
    text_cases = []
    for text in TOY_TEXTS:
        ids = tok(text, truncation=True, max_length=TOY_COMMON["max_seq"])["input_ids"]
        text_cases.append({"text": text, "ids": ids, "vector": toy_forward_expected(bert, ids, "cls")})
    expected["packs"]["toy-absolute-gelu"] = {
        "pooling": "cls",
        "forward_cases": [
            {"ids": ids, "vector": toy_forward_expected(bert, ids, "cls")} for ids in TOY_ID_SEQS
        ],
        "text_cases": text_cases,
    }

    # ── rope + swiglu + mean (+ matryoshka recipe at dim 16) ──
    pack = toy_dir / "toy-rope-swiglu"
    pack.mkdir(parents=True, exist_ok=True)
    tensors, nomic = build_toy_nomic()
    write_safetensors(tensors, pack / "model.safetensors", "F32")
    cfg = toy_pack_config("toy-rope-swiglu", "rope", "swiglu", "mean", matryoshka=True)
    (pack / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (pack / "vocab.txt").write_text("\n".join(TOY_VOCAB) + "\n", encoding="utf-8", newline="\n")
    write_toy_manifest(pack, "toy-rope-swiglu", list(tensors))
    expected["packs"]["toy-rope-swiglu"] = {
        "pooling": "mean",
        "forward_cases": [
            {"ids": ids, "vector": toy_forward_expected(nomic, ids, "mean")} for ids in TOY_ID_SEQS
        ],
        "matryoshka_cases": [
            {"ids": ids, "dim": 16, "vector": toy_forward_expected(nomic, ids, "mean", matryoshka_dim=16)}
            for ids in TOY_ID_SEQS[:2]
        ],
    }

    path = toy_dir / "expected.json"
    path.write_text(json.dumps(expected, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {toy_dir}")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["all", "tokenizer", "toys", "parity"], default="all")
    # Parity-only model filter (mirrors build-weight-pack.py --model), so a
    # new pack's goldens can be generated without re-running — and therefore
    # without risking churn in — the committed goldens of the others.
    ap.add_argument("--model", choices=["all", "nomic", "bge", "arctic"], default="all")
    args = ap.parse_args()
    model_ids = {
        "all": "all",
        "nomic": "nomic-embed-text-v1.5",
        "bge": "bge-small-en-v1.5",
        "arctic": "snowflake-arctic-embed-m-long",
    }
    TESTDATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.only in ("all", "tokenizer"):
        gen_tokenizer_fixtures()
    if args.only in ("all", "toys"):
        gen_toys()
    if args.only in ("all", "parity"):
        gen_parity_fixtures(model_ids[args.model])
    print("OK")


if __name__ == "__main__":
    main()
