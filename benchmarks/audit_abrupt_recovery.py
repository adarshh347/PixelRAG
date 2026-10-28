"""Kill only this experiment's child processes at publication boundaries."""

import hashlib
import json
import os
import signal
import subprocess
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
OPTIONS = dict(nlist=2, nprobe=2, train_sample=128, batch_size=32)
if len(sys.argv) > 2 and sys.argv[1] == "worker":
    root = Path(sys.argv[2])
    boundary = sys.argv[3]
    rename = s.os.rename

    def interrupt(src, dst):
        if boundary == "after":
            rename(src, dst)
        os.kill(os.getpid(), signal.SIGKILL)

    with patch.object(s.os, "rename", interrupt):
        s.build_streaming_ivf(root / "shards", root / "index", **OPTIONS)
    raise AssertionError("interruption missed")
report = []
with tempfile.TemporaryDirectory(prefix="pixelrag-abrupt-audit-") as tmp:
    for boundary in ("before", "after"):
        root = Path(tmp) / boundary
        root.mkdir()
        generate_shards(root / "shards", rows=512, dim=16, shard_rows=128)
        child = subprocess.run(
            [sys.executable, __file__, "worker", str(root), boundary], timeout=30
        )
        assert child.returncode == -signal.SIGKILL
        output = root / "index"
        published = output.exists()
        assert published == (boundary == "after")
        orphans = list(root.glob(".ivf-study-*"))
        assert len(orphans) == 1
        if not published:
            s.build_streaming_ivf(root / "shards", output, **OPTIONS)
            retry = "successful clean rebuild; orphan retained"
        else:
            before = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
            }
            try:
                s.build_streaming_ivf(root / "shards", output, **OPTIONS)
            except FileExistsError:
                pass
            else:
                raise AssertionError("existing output overwritten")
            assert before == {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.iterdir()
            }
            retry = "existing published output rejected and unchanged"
        index = faiss.read_index(str(output / "index.faiss"))
        assert index.ntotal == 487
        with np.load(output / "metadata.npz") as data:
            assert all(len(data[f]) == 487 for f in s.FIELDS)
        assert json.loads((output / "summary.json").read_text())["total_vectors"] == 487
        report.append(
            dict(
                boundary=boundary,
                signal="SIGKILL",
                published_at_death=published,
                orphan_directories=len(orphans),
                retry=retry,
                readable_vectors=487,
            )
        )
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
