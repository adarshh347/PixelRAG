"""Injected exception and retry audit; never fills disk or kills real services."""

import errno
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

from benchmarks import streaming_ivf as s
from benchmarks.index_build_memory import generate_shards

faiss.omp_set_num_threads(1)
report = []
with tempfile.TemporaryDirectory(prefix="pixelrag-recovery-") as tmp:
    root = Path(tmp)
    generate_shards(root / "shards", rows=512, dim=16, shard_rows=128)
    options = dict(nlist=2, nprobe=2, train_sample=128, batch_size=32)
    for stage in ("metadata", "index", "summary", "rename"):
        output = root / stage

        def fail(*args, **kwargs):
            raise OSError(errno.ENOSPC, "injected disk-full exception")

        original_write = Path.write_text

        def write(path, *args, original_write=original_write, **kwargs):
            if path.name == "summary.json":
                return fail()
            return original_write(path, *args, **kwargs)

        injection = {
            "metadata": lambda: patch.object(s, "_pack_metadata", fail),
            "index": lambda: patch.object(faiss, "write_index", fail),
            "summary": lambda: patch.object(Path, "write_text", write),
            "rename": lambda: patch.object(s.os, "rename", fail),
        }[stage]
        with injection():
            try:
                s.build_streaming_ivf(root / "shards", output, **options)
            except OSError as error:
                assert error.errno == errno.ENOSPC
            else:
                raise AssertionError("injection not reached")
        assert not output.exists()
        assert not list(root.glob(".ivf-study-*"))
        s.build_streaming_ivf(root / "shards", output, **options)
        index = faiss.read_index(str(output / "index.faiss"))
        with np.load(output / "metadata.npz") as meta:
            assert all(len(meta[field]) == index.ntotal for field in s.FIELDS)
        assert index.ntotal == 487
        report.append(
            dict(
                stage=stage,
                partial_destination=False,
                staging_cleaned=True,
                retry_vectors=index.ntotal,
            )
        )
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
