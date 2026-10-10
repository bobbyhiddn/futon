#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "sentence-transformers",
#   "numpy",
# ]
# ///
"""Check a futon weight pack against the reference implementation.

Embeds a fixed set of texts (prose, code, accents, CJK, an empty string and an
over-long input) with futon and with sentence-transformers, and reports the
cosine similarity per text. A correct pack scores >= 0.9999 on every text.

Usage:
    scripts/check-pack.py --repo sentence-transformers/all-MiniLM-L6-v2 \\
        --revision <40-char sha> --pack-id all-minilm-l6-v2 \\
        [--models-dir ~/.futon/models] [--futon futon] [--no-normalize]

--futon is the futon command (build it with `go build ./cmd/futon`). Only
models that sentence-transformers can load are supported; for a bare BERT
checkpoint without a sentence-transformers config, compare against your own
reference instead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "How does the recall pipeline decide between full-text and vector search?",
    "Élodie Müller façade naïve coöperate señor café crème brûlée",
    "def cosine(a, b):\n    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))",
    "自然言語処理は面白い。 机器学习模型需要大量数据。",
    "",
    "word " * 600,
]
GATE = 0.9999


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--pack-id", required=True)
    ap.add_argument("--models-dir", default=str(Path.home() / ".futon" / "models"))
    ap.add_argument("--futon", default="futon", help="path to the futon command")
    ap.add_argument("--no-normalize", action="store_true", help="the model does not L2-normalize its output")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.repo, revision=args.revision, device="cpu")
    ref = model.encode(TEXTS, normalize_embeddings=not args.no_normalize)

    out = subprocess.run(
        [args.futon, "embed", "-model", args.pack_id, "-models-dir", args.models_dir, *TEXTS],
        capture_output=True, text=True, check=True, encoding="utf-8",
    ).stdout.strip().split("\n")
    got = np.array([json.loads(line) for line in out], dtype=np.float32)

    worst = 1.0
    for text, a, b in zip(TEXTS, ref, got):
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) or 1.0))
        worst = min(worst, cos)
        label = text[:48].replace("\n", " ")
        print(f"{cos:.6f}  {label!r}".encode("utf-8", "replace").decode("utf-8"))
    print(f"min cosine {worst:.6f}  ({'PASS' if worst >= GATE else 'FAIL'}, gate {GATE})")
    return 0 if worst >= GATE else 1


if __name__ == "__main__":
    sys.exit(main())
