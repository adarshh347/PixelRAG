"""Correctness-only control: replace training inputs, preserve both build paths.

Run with the memory-study dev interpreter. No performance claims: the oracle
and common training matrix are deliberately held in this process's memory.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "embed/src"))
import faiss
import numpy as np
from pixelrag_embed.index import _merge_all_shards, build_ivf

from benchmarks.index_build_memory import generate_shards
from benchmarks.streaming_ivf import FIELDS, build_streaming_ivf

faiss.omp_set_num_threads(1)
report = {
    "kind": "correctness-only matched-training control",
    "numpy": np.__version__,
    "faiss": faiss.__version__,
    "cases": [],
}
with tempfile.TemporaryDirectory(prefix="pixelrag-matched-training-") as temp:
    root = Path(temp)
    generate_shards(root / "shards", rows=32768, dim=64, shard_rows=4096, seed=2026)
    merged = _merge_all_shards(sorted((root / "shards").glob("*.npz")))
    vectors = merged["embeddings"]
    rng = np.random.default_rng(90210)
    common = np.ascontiguousarray(
        vectors[rng.choice(len(vectors), 4096, replace=False)]
    )
    digest = hashlib.sha256(common.tobytes()).hexdigest()
    queries = rng.standard_normal((128, 64)).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    for pq in (0, 8):
        calls = []
        cls = faiss.IndexIVFPQ if pq else faiss.IndexIVFFlat
        original_train = cls.train

        def train(index, supplied, calls=calls, original_train=original_train):
            calls.append(
                {
                    "supplied_sha256": hashlib.sha256(supplied.tobytes()).hexdigest(),
                    "used_sha256": digest,
                }
            )
            return original_train(index, common)

        options = dict(nlist=32, nprobe=8, train_sample=4096, pq_m=pq, pq_nbits=4)
        left = root / f"legacy-{pq}"
        right = root / f"prototype-{pq}"
        with (
            patch.object(cls, "train", train),
            patch("pixelrag_embed.index.os.cpu_count", return_value=1),
        ):
            np.random.seed(2026)
            build_ivf(str(root / "shards"), str(left), **options)
            build_streaming_ivf(
                root / "shards", right, seed=2026, batch_size=1024, **options
            )
        assert len(calls) == 2 and calls[0]["used_sha256"] == calls[1]["used_sha256"]
        a = faiss.read_index(str(left / "index.faiss"))
        b = faiss.read_index(str(right / "index.faiss"))
        assert a.ntotal == b.ntotal == len(vectors)
        with (
            np.load(left / "metadata.npz") as ma,
            np.load(right / "metadata.npz") as mb,
        ):
            for field in FIELDS:
                np.testing.assert_array_equal(ma[field], merged[field])
                np.testing.assert_array_equal(mb[field], merged[field])
        probes = []
        for nprobe in (8, 32):
            a.nprobe = b.nprobe = nprobe
            da, ia = a.search(queries, 10)
            db, ib = b.search(queries, 10)
            np.testing.assert_array_equal(ia, ib)
            np.testing.assert_allclose(da, db, rtol=1e-6, atol=1e-6)
            probes.append(
                {
                    "nprobe": nprobe,
                    "same_ids": True,
                    "max_score_difference": float(np.max(np.abs(da - db))),
                }
            )
        report["cases"].append(
            {
                "pq_m": pq,
                "raw_rows": 32768,
                "unique_rows": len(vectors),
                "dimension": 64,
                "training_rows": 4096,
                "queries": 128,
                "training_calls": calls,
                "metadata_equal": True,
                "probes": probes,
            }
        )
output = Path(sys.argv[1])
output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
