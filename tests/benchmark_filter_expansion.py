"""Local fixed-vector comparison with the audited API (no embedding/rendering).

uv run --no-sync python tests/benchmark_filter_expansion.py --iterations 100

Loads the baseline API from git without modifying the checkout. Both policies
use the same real flat FAISS / in-memory Qdrant corpus and fixed query vector.
Reports median endpoint latency, raw_search calls, requested candidate slots
(including repeated prefixes), returned candidates, and eligible hits.
"""

import argparse
import csv
import importlib.util
import logging
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from pixelrag_serve import api
from pixelrag_serve.backends import FaissBackend, QdrantBackend


class CountingBackend:
    def __init__(self, backend):
        self.backend = backend
        self.reset_counts()

    def reset_counts(self):
        self.calls = self.requested = self.returned = 0

    def set_nprobe(self, n):
        self.backend.set_nprobe(n)

    def reset_nprobe(self):
        self.backend.reset_nprobe()

    def raw_search(self, vectors, k, **kwargs):
        self.calls += 1
        self.requested += len(vectors) * k
        rows = self.backend.raw_search(vectors, k, **kwargs)
        self.returned += sum(map(len, rows))
        return rows


def make_backends(root):
    # Strictly decreasing dot products with [1, 0, 0, 0]; no ties or RNG.
    x = np.linspace(0.99, 0.1, 80, dtype=np.float32)
    vectors = np.column_stack((x, np.sqrt(1 - x * x), np.zeros((80, 2)))).astype(
        np.float32
    )
    index = faiss.IndexFlatIP(4)
    index.add(vectors)
    faiss.write_index(index, str(root / "index.faiss"))
    np.savez(
        root / "metadata.npz",
        article_ids=np.arange(80),
        tile_indices=np.zeros(80, dtype=int),
        chunk_indices=np.zeros(80, dtype=int),
        y_offsets=np.zeros(80, dtype=int),
        tile_heights=np.full(80, 100),
    )
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        "pixelrag",
        vectors_config=models.VectorParams(size=4, distance=models.Distance.DOT),
    )
    qdrant.upsert(
        "pixelrag",
        points=[
            models.PointStruct(
                id=i,
                vector=v.tolist(),
                payload={
                    "article_id": i,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 100,
                },
            )
            for i, v in enumerate(vectors)
        ],
    )
    qb = QdrantBackend.__new__(QdrantBackend)
    qb.client, qb.collection, qb.dimension = qdrant, "pixelrag", 4
    return {"faiss-flat": FaissBackend(str(root), {"dimension": 4}), "qdrant-local": qb}


def load_baseline(root, ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:serve/src/pixelrag_serve/api.py"]
    )
    path = root / "baseline_api.py"
    path.write_bytes(source)
    name = "pixelrag_serve._benchmark_baseline_api"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure(module, backend, root, meta_count, batch, articles_only, iterations):
    counted = CountingBackend(backend)
    old_state = module._state
    module._state = {
        "backend": counted,
        "dimension": 4,
        "tiles_dir": str(root / "empty"),
        "articles": [
            f"Portal:Meta_{i}" if i < meta_count else f"Article_{i}" for i in range(80)
        ],
    }
    body = {
        "queries": [{"embedding": [1, 0, 0, 0]}] * batch,
        "n_docs": 2,
        "articles_only": articles_only,
    }
    times, counts = [], []
    module._article_pages.cache_clear()
    try:
        with TestClient(module.app) as client:
            for i in range(iterations + 10):
                counted.reset_counts()
                start = time.perf_counter()
                response = client.post("/search", json=body)
                elapsed = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                hits = sum(len(r["hits"]) for r in response.json()["results"])
                if i >= 10:
                    times.append(elapsed)
                    counts.append(
                        (counted.calls, counted.requested, counted.returned, hits)
                    )
        assert len(set(counts)) == 1, "fixed-vector results changed between repetitions"
        return (*counts[0], round(statistics.median(times), 3))
    finally:
        module._state = old_state
        module._article_pages.cache_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--baseline-ref", default="b4fe76301465eaee3d8ac75e0bec54664e53c072"
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    logging.disable(logging.CRITICAL)
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "backend",
            "case",
            "policy",
            "calls",
            "requested",
            "returned",
            "hits",
            "median_ms",
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        baseline = load_baseline(root, args.baseline_ref)
        backends = make_backends(root)
        try:
            for name, backend in backends.items():
                for case, meta, batch, filtered in [
                    ("unfiltered", 0, 1, False),
                    ("first_suffices", 0, 1, True),
                    ("beyond_20", 20, 1, True),
                    ("no_match", 80, 1, True),
                    ("cap_before_match", 40, 1, True),
                    ("batch_3", 20, 3, True),
                ]:
                    for policy, module in [("original", baseline), ("proposed", api)]:
                        writer.writerow(
                            [
                                name,
                                case,
                                policy,
                                *measure(
                                    module,
                                    backend,
                                    root,
                                    meta,
                                    batch,
                                    filtered,
                                    args.iterations,
                                ),
                            ]
                        )
        finally:
            backends["qdrant-local"].client.close()


if __name__ == "__main__":
    main()
