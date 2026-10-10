#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface_hub==0.34.4",
#   "safetensors==0.4.5",
#   "numpy>=1.26,<3",
#   "zstandard==0.23.0",
# ]
# ///
"""Build futon native-embedding weight packs from pinned upstream revisions.

This is the seed of the Phase-2 CI packaging job: deterministic (same inputs
-> byte-identical pack) and self-verifying (the pack is re-read and checked
against the manifest and against the source tensors before the script exits 0).

For each model it:
  1. downloads model.safetensors, vocab.txt, README.md at a PINNED revision
     (commit hashes recorded below — never a branch name),
  2. converts/renames tensors to the canonical futon pack scheme, cast to
     fp16 (little-endian, sorted tensor names, minimal sorted-key JSON header,
     header padded to 8-byte alignment => byte-reproducible),
  3. writes config.json (the pack architecture description Go consumes),
     vocab.txt, LICENSE, and manifest.json with per-member sha256.

Usage:
    scripts/build-weight-pack.py [--models-dir ~/.futon/models] [--model all|nomic|bge]
    scripts/build-weight-pack.py --archive [--archive-dir dist/weights]
    scripts/build-weight-pack.py --archive-existing [--archive-dir dist/weights]

--archive additionally packages each built pack as a `.futonw` weight-pack
archive (ARCHIVE_SUFFIX; a DETERMINISTIC tar.zst of the pack members: sorted
entries, fixed mtime/owner/mode, single-thread zstd-19 — re-runs are
byte-identical) and writes/updates weights-checksums.txt (sha256sum format,
one line per archive).
--archive-existing archives packs already present in --models-dir without
any download or rebuild (fully offline).

No GPU, no torch, no trust_remote_code: only safetensors + numpy (+ zstandard
for --archive).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# ── Pinned upstream revisions (commit SHAs on huggingface.co) ────────────────
# Verified 2026-08-16 (nomic/bge) and 2026-08-17 (arctic) via the HF API.
# Bumping a pin is a reviewed change: it moves the vector space and must
# re-run the golden parity gate.
NOMIC_REPO = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
BGE_REPO = "BAAI/bge-small-en-v1.5"
BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
ARCTIC_REPO = "Snowflake/snowflake-arctic-embed-m-long"
ARCTIC_REVISION = "92d97331f1f4b6a366c1f161354b9f3390cc219f"

PACK_FORMAT = 1

# Upstream repos ship no LICENSE file; the license is declared in the model
# card front matter. We verify the declaration matches what we expect, then
# write a LICENSE file recording identifier + provenance.
EXPECTED_LICENSE = {NOMIC_REPO: "apache-2.0", BGE_REPO: "mit", ARCTIC_REPO: "apache-2.0"}

# LOW-4: both launch models ship the identical bert-base-uncased vocab.txt,
# and internal/embed/native/testdata/vocab_bert_uncased.txt is committed as
# that exact file so tokenizer conformance runs in CI. This recorded hash is
# mirrored by expectedVocabSHA256 in tokenizer_test.go — if an upstream pin
# bump ever changes the vocab, BOTH must change in one reviewed commit
# (and the committed test vocab must be replaced), or the build fails here.
EXPECTED_VOCAB_SHA256 = "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"

NOMIC_CONFIG = {
    "pack_format": PACK_FORMAT,
    "model_id": "nomic-embed-text-v1.5",
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 30528,
    "type_vocab_size": 2,
    "max_seq_len": 8192,
    "layer_norm_eps": 1e-12,
    "position_encoding": "rope",
    "rope_theta": 1000.0,
    "max_trained_positions": 2048,
    "activation": "swiglu",
    "pooling": "mean",
    "normalize": True,
    "matryoshka": {"layernorm": True, "dims": [768, 512, 256, 128, 64]},
    "tokenizer": {
        "do_lower_case": True,
        "tokenize_chinese_chars": True,
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
    },
    # Provenance only. futon embeds raw text; the provider never applies these.
    "prefixes": {
        "search_query": "search_query: ",
        "search_document": "search_document: ",
        "classification": "classification: ",
        "clustering": "clustering: ",
    },
}

# snowflake-arctic-embed-m-long is nomic-bert architecture (NomicBertModel,
# RoPE θ=1000 + SwiGLU, no attention/FFN biases, post-LN) with CLS pooling.
# Upstream declares 8192 positions via DYNAMIC-NTK RoPE (rotary_scaling_factor
# 2) — the NTK rescale only engages for seqlen > max_trained_positions (2048)
# in the pinned modeling code, so within the trained context the model is
# plain RoPE base 1000, exactly what the Go engine implements. The pack pins
# max_trained_positions 2048 and the loader's trained-context clamp keeps
# every input inside that regime (NTK scaling for 2049..8192 is future work,
# same as nomic's).
ARCTIC_CONFIG = {
    "pack_format": PACK_FORMAT,
    "model_id": "snowflake-arctic-embed-m-long",
    "hidden_size": 768,
    "num_layers": 12,
    "num_heads": 12,
    "intermediate_size": 3072,
    "vocab_size": 30528,
    "type_vocab_size": 2,
    "max_seq_len": 8192,
    "layer_norm_eps": 1e-12,
    "position_encoding": "rope",
    "rope_theta": 1000.0,
    "max_trained_positions": 2048,
    "activation": "swiglu",
    "pooling": "cls",
    "normalize": True,
    "tokenizer": {
        "do_lower_case": True,
        "tokenize_chinese_chars": True,
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
    },
    # Provenance only. futon embeds raw text; the provider never applies
    # this (arctic prefixes queries only; documents are embedded raw).
    "prefixes": {
        "query": "Represent this sentence for searching relevant passages: ",
    },
}

BGE_CONFIG = {
    "pack_format": PACK_FORMAT,
    "model_id": "bge-small-en-v1.5",
    "hidden_size": 384,
    "num_layers": 12,
    "num_heads": 12,
    "intermediate_size": 1536,
    "vocab_size": 30522,
    "type_vocab_size": 2,
    "max_seq_len": 512,
    "layer_norm_eps": 1e-12,
    "position_encoding": "absolute",
    "activation": "gelu",
    "pooling": "cls",
    "normalize": True,
    "tokenizer": {
        "do_lower_case": True,
        "tokenize_chinese_chars": True,
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_safetensors_fp16(tensors: dict[str, np.ndarray], out: Path) -> None:
    """Deterministic minimal safetensors writer: fp16, sorted names, compact
    sorted-key JSON, header space-padded to 8-byte alignment (as the reference
    implementation does)."""
    names = sorted(tensors)
    header: dict[str, dict] = {}
    offset = 0
    blobs: list[bytes] = []
    for name in names:
        arr = tensors[name]
        assert arr.dtype == np.float16, f"{name}: expected fp16, got {arr.dtype}"
        blob = np.ascontiguousarray(arr).astype("<f2", copy=False).tobytes()
        header[name] = {
            "dtype": "F16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        blobs.append(blob)
    hjson = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    pad = (8 - (8 + len(hjson)) % 8) % 8
    hjson += b" " * pad
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for blob in blobs:
            f.write(blob)


def load_source_tensors(st_path: Path) -> dict[str, np.ndarray]:
    out = {}
    with safe_open(st_path, framework="numpy") as f:
        for name in f.keys():
            out[name] = f.get_tensor(name)
    return out


def canonicalize_nomic(src: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """nomic-bert (NomicBertModel) names -> canonical pack names.

    Wqkv rows are already (three, heads, head_dim)-major — q block first,
    then k, then v — exactly the canonical packed layout. fc11 is the value/up
    projection, fc12 the silu gate (forward: y = fc11(x) * silu(fc12(x)))."""
    n = cfg["num_layers"]
    mapping = {
        "embeddings.word_embeddings.weight": "embeddings.word",
        "embeddings.token_type_embeddings.weight": "embeddings.token_type",
        "emb_ln.weight": "embeddings.norm.weight",
        "emb_ln.bias": "embeddings.norm.bias",
    }
    for i in range(n):
        s, d = f"encoder.layers.{i}", f"layers.{i}"
        mapping.update(
            {
                f"{s}.attn.Wqkv.weight": f"{d}.attn.qkv.weight",
                f"{s}.attn.out_proj.weight": f"{d}.attn.out.weight",
                f"{s}.norm1.weight": f"{d}.norm1.weight",
                f"{s}.norm1.bias": f"{d}.norm1.bias",
                f"{s}.mlp.fc11.weight": f"{d}.mlp.fc1.weight",
                f"{s}.mlp.fc12.weight": f"{d}.mlp.gate.weight",
                f"{s}.mlp.fc2.weight": f"{d}.mlp.fc2.weight",
                f"{s}.norm2.weight": f"{d}.norm2.weight",
                f"{s}.norm2.bias": f"{d}.norm2.bias",
            }
        )
    missing = [k for k in mapping if k not in src]
    if missing:
        raise SystemExit(f"nomic source missing expected tensors: {missing[:5]}")
    extra = [k for k in src if k not in mapping]
    if extra:
        raise SystemExit(f"nomic source has unexpected tensors (arch drift?): {extra[:5]}")
    return {dst: src[s].astype(np.float16) for s, dst in mapping.items()}


def canonicalize_bge(src: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """HF BertModel names -> canonical pack names. Separate q/k/v are packed
    into one [3h, h] qkv tensor (q rows, then k, then v); the unused pooler
    head is dropped."""
    n = cfg["num_layers"]
    out: dict[str, np.ndarray] = {}
    used = set()

    def take(name: str) -> np.ndarray:
        if name not in src:
            raise SystemExit(f"bge source missing expected tensor: {name}")
        used.add(name)
        return src[name]

    out["embeddings.word"] = take("embeddings.word_embeddings.weight")
    out["embeddings.position"] = take("embeddings.position_embeddings.weight")
    out["embeddings.token_type"] = take("embeddings.token_type_embeddings.weight")
    out["embeddings.norm.weight"] = take("embeddings.LayerNorm.weight")
    out["embeddings.norm.bias"] = take("embeddings.LayerNorm.bias")
    for i in range(n):
        s, d = f"encoder.layer.{i}", f"layers.{i}"
        for kind in ("weight", "bias"):
            q = take(f"{s}.attention.self.query.{kind}")
            k = take(f"{s}.attention.self.key.{kind}")
            v = take(f"{s}.attention.self.value.{kind}")
            out[f"{d}.attn.qkv.{kind}"] = np.concatenate([q, k, v], axis=0)
            out[f"{d}.attn.out.{kind}"] = take(f"{s}.attention.output.dense.{kind}")
            out[f"{d}.norm1.{kind}"] = take(f"{s}.attention.output.LayerNorm.{kind}")
            out[f"{d}.mlp.fc1.{kind}"] = take(f"{s}.intermediate.dense.{kind}")
            out[f"{d}.mlp.fc2.{kind}"] = take(f"{s}.output.dense.{kind}")
            out[f"{d}.norm2.{kind}"] = take(f"{s}.output.LayerNorm.{kind}")
    # Pooler head (unused by CLS-pool sentence embedding) and the registered
    # position_ids buffer (just arange(512)) are deliberately dropped.
    for name in ("pooler.dense.weight", "pooler.dense.bias", "embeddings.position_ids"):
        used.add(name)
    extra = [k for k in src if k not in used]
    if extra:
        raise SystemExit(f"bge source has unexpected tensors (arch drift?): {extra[:5]}")
    return {k: v.astype(np.float16) for k, v in out.items()}


def declared_license(readme: Path) -> str:
    """Extract `license: X` from the model card YAML front matter."""
    text = readme.read_text(encoding="utf-8")
    m = re.search(r"^license:\s*(\S+)\s*$", text, flags=re.M)
    if not m:
        raise SystemExit(f"{readme}: no license declaration in model card")
    return m.group(1)


SPDX_IDS = {"mit": "MIT", "apache-2.0": "Apache-2.0"}


def write_license(dst: Path, repo: str, revision: str, license_id: str) -> None:
    dst.write_text(
        f"Model weights license: {license_id}\n"
        f"Upstream: https://huggingface.co/{repo}\n"
        f"Revision: {revision}\n\n"
        f"The upstream repository declares its license in the model card front\n"
        f"matter and ships no standalone LICENSE file. Canonical license text:\n"
        f"https://spdx.org/licenses/{SPDX_IDS[license_id]}.html\n",
        encoding="utf-8",
        newline="\n",
    )


def verify_pack(pack_dir: Path, cfg: dict, source: dict[str, np.ndarray], canonical: dict[str, np.ndarray]) -> None:
    """Self-verification: manifest hashes match the files on disk, the written
    safetensors round-trips to exactly the canonical fp16 values, and the
    vocab has the expected shape."""
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    for member, meta in manifest["members"].items():
        p = pack_dir / member
        got = sha256_file(p)
        if got != meta["sha256"]:
            raise SystemExit(f"verify: {member}: sha256 mismatch ({got} != {meta['sha256']})")
        if p.stat().st_size != meta["size"]:
            raise SystemExit(f"verify: {member}: size mismatch")
    reread = load_source_tensors(pack_dir / "model.safetensors")
    if set(reread) != set(canonical):
        raise SystemExit("verify: tensor name set mismatch after round-trip")
    for name, arr in canonical.items():
        back = reread[name]
        if back.dtype != np.float16 or back.shape != arr.shape or not np.array_equal(back, arr):
            raise SystemExit(f"verify: tensor {name} did not round-trip bit-exactly")
    vocab_lines = (pack_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if len(vocab_lines) > cfg["vocab_size"]:
        raise SystemExit(f"verify: vocab has {len(vocab_lines)} entries > vocab_size {cfg['vocab_size']}")
    for tok in ("[CLS]", "[SEP]", "[UNK]", "[PAD]"):
        if tok not in vocab_lines:
            raise SystemExit(f"verify: vocab missing special token {tok}")
    # Spot-check dequantization drift bound: fp16 cast must be the only loss.
    name = "embeddings.word"
    src32 = source_word_embedding(source)
    drift = np.abs(src32.astype(np.float32) - canonical[name].astype(np.float32)).max()
    if drift > 0.005:
        raise SystemExit(f"verify: fp16 cast drift {drift} on {name} exceeds bound")
    print(f"  verified: {len(manifest['members'])} members, {len(reread)} tensors round-trip bit-exact")


def source_word_embedding(source: dict[str, np.ndarray]) -> np.ndarray:
    return source["embeddings.word_embeddings.weight"]


def build_pack(repo: str, revision: str, cfg: dict, models_dir: Path) -> None:
    pack_dir = models_dir / cfg["model_id"]
    pack_dir.mkdir(parents=True, exist_ok=True)
    print(f"== {cfg['model_id']}  ({repo} @ {revision[:12]})")

    st_path = Path(hf_hub_download(repo, "model.safetensors", revision=revision))
    vocab_path = Path(hf_hub_download(repo, "vocab.txt", revision=revision))
    readme_path = Path(hf_hub_download(repo, "README.md", revision=revision))

    license_id = declared_license(readme_path)
    if license_id != EXPECTED_LICENSE[repo]:
        raise SystemExit(f"{repo}: model card declares license {license_id!r}, expected {EXPECTED_LICENSE[repo]!r}")

    vocab_sha = sha256_file(vocab_path)
    if vocab_sha != EXPECTED_VOCAB_SHA256:
        raise SystemExit(
            f"{repo}: vocab.txt sha256 {vocab_sha} != recorded {EXPECTED_VOCAB_SHA256}.\n"
            "The committed test vocab (internal/embed/native/testdata/vocab_bert_uncased.txt)\n"
            "and tokenizer conformance fixtures would no longer describe this pack. Update the\n"
            "recorded hash here AND expectedVocabSHA256 in tokenizer_test.go AND the committed\n"
            "vocab + fixtures in one reviewed change."
        )

    source = load_source_tensors(st_path)
    if cfg["model_id"] in ("nomic-embed-text-v1.5", "snowflake-arctic-embed-m-long"):
        canonical = canonicalize_nomic(source, cfg)
    else:
        canonical = canonicalize_bge(source, cfg)

    write_safetensors_fp16(canonical, pack_dir / "model.safetensors")
    (pack_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    (pack_dir / "vocab.txt").write_bytes(vocab_path.read_bytes())
    write_license(pack_dir / "LICENSE", repo, revision, license_id)

    members = {}
    for member in ("config.json", "model.safetensors", "vocab.txt", "LICENSE"):
        p = pack_dir / member
        members[member] = {"sha256": sha256_file(p), "size": p.stat().st_size}
    manifest = {
        "pack_format": PACK_FORMAT,
        "model_id": cfg["model_id"],
        "created_by": "scripts/build-weight-pack.py",
        # Both sequence-length truths travel with the pack: what upstream
        # DECLARES (max_seq_len) and what the model was TRAINED on. The Go
        # loader clamps to the trained value — plain RoPE extrapolation past
        # it is a measured silent-degradation regime (see
        # docs/design/native-embeddings.md).
        "declared_max_seq_len": cfg["max_seq_len"],
        "trained_seq_len": cfg.get("max_trained_positions", cfg["max_seq_len"]),
        # The exact tensor-name set (optionals included). The loader verifies
        # the loaded set equals this list, so deleting an optional tensor
        # (e.g. a bias) cannot silently change the function.
        "tensor_names": sorted(canonical),
        "upstream": {
            "repo": repo,
            "revision": revision,
            "source_file": "model.safetensors",
            "source_sha256": sha256_file(st_path),
            "license": license_id,
        },
        "members": members,
    }
    (pack_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )

    verify_pack(pack_dir, cfg, source, canonical)
    total = sum(m["size"] for m in members.values())
    print(f"  pack: {pack_dir}  ({total / 1e6:.1f} MB)")


# ── weight-pack distribution archives ─────────────────────────────────────────
# Format: tar.zst of the pack directory members, FLAT (no directory prefix):
# config.json, model.safetensors, vocab.txt, manifest.json, LICENSE.
# Consumed by internal/modelpack (Resolve/Install), which re-verifies every
# member against manifest.json before an atomic install.
#
# Determinism contract (release assets are append-only; a re-run must
# reproduce the hosted bytes exactly or fail the byte-compare in CI):
#   - entries sorted by name
#   - mtime=0, uid=gid=0, empty uname/gname, mode 0644, USTAR format
#   - single zstd frame, level 19, single thread, no content-size surprises
# The archive extension. internal/modelpack.ArchiveSuffix is the source of
# truth (TestArchiveSuffixIsSpelledOnce keeps this copy in step with it).
ARCHIVE_SUFFIX = ".futonw"
ARCHIVE_MEMBERS = ("config.json", "model.safetensors", "vocab.txt", "manifest.json", "LICENSE")
ZSTD_LEVEL = 19
CHECKSUMS_NAME = "weights-checksums.txt"


def write_pack_archive(pack_dir: Path, out_path: Path) -> str:
    """Archive one pack directory as a deterministic weight-pack archive. Returns sha256."""
    import io
    import tarfile

    import zstandard

    for member in ARCHIVE_MEMBERS:
        if not (pack_dir / member).is_file():
            raise SystemExit(f"archive: {pack_dir} is missing member {member}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for member in sorted(ARCHIVE_MEMBERS):
            p = pack_dir / member
            info = tarfile.TarInfo(name=member)
            info.size = p.stat().st_size
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with p.open("rb") as f:
                tf.addfile(info, f)

    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=0, write_checksum=True)
    compressed = cctx.compress(buf.getvalue())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(compressed)
    sha = hashlib.sha256(compressed).hexdigest()
    print(f"  archive: {out_path}  ({len(compressed) / 1e6:.1f} MB)  sha256 {sha[:16]}…")
    return sha


def update_checksums(archive_dir: Path, entries: dict[str, str]) -> None:
    """Merge entries into weights-checksums.txt (sha256sum format, sorted)."""
    path = archive_dir / CHECKSUMS_NAME
    existing: dict[str, str] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2 and fields[1].endswith(ARCHIVE_SUFFIX):
                existing[fields[1]] = fields[0]
    existing.update(entries)
    lines = [f"{sha}  {name}" for name, sha in sorted(existing.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"  checksums: {path}  ({len(existing)} entries)")


def archive_pack(models_dir: Path, archive_dir: Path, model_id: str) -> tuple[str, str]:
    pack_dir = models_dir / model_id
    if not (pack_dir / "manifest.json").is_file():
        raise SystemExit(f"archive: no pack at {pack_dir} (build it first, or drop --archive-existing)")
    name = f"{model_id}{ARCHIVE_SUFFIX}"
    sha = write_pack_archive(pack_dir, archive_dir / name)
    return name, sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-dir", default=str(Path.home() / ".futon" / "models"))
    ap.add_argument("--model", choices=["all", "nomic", "bge", "arctic"], default="all")
    ap.add_argument("--archive", action="store_true",
                    help=f"after building, write <pack-id>{ARCHIVE_SUFFIX} distribution archives + weights-checksums.txt")
    ap.add_argument("--archive-existing", action="store_true",
                    help="archive packs already present in --models-dir (no download, no rebuild)")
    ap.add_argument("--archive-dir", default="dist/weights",
                    help=f"output directory for {ARCHIVE_SUFFIX} archives and weights-checksums.txt")
    args = ap.parse_args()
    models_dir = Path(args.models_dir).expanduser()
    archive_dir = Path(args.archive_dir).expanduser()

    selected = []
    if args.model in ("all", "nomic"):
        selected.append((NOMIC_REPO, NOMIC_REVISION, NOMIC_CONFIG))
    if args.model in ("all", "bge"):
        selected.append((BGE_REPO, BGE_REVISION, BGE_CONFIG))
    if args.model in ("all", "arctic"):
        selected.append((ARCTIC_REPO, ARCTIC_REVISION, ARCTIC_CONFIG))

    if not args.archive_existing:
        for repo, revision, cfg in selected:
            build_pack(repo, revision, cfg, models_dir)

    if args.archive or args.archive_existing:
        checksums: dict[str, str] = {}
        for _, _, cfg in selected:
            name, sha = archive_pack(models_dir, archive_dir, cfg["model_id"])
            checksums[name] = sha
        update_checksums(archive_dir, checksums)
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
