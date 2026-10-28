# Streaming index evidence package

Experimental evidence for upstream issue #179, not production integration.
The branch starts at the reviewed prototype e9688df (production baseline b4fe763).
Current upstream cf91bd4 has no changes to the production embed index builder
relative to that baseline. Production files, defaults and lockfile are unchanged.

## Reproduce from this checkout

```
uv sync --locked --extra dev --python 3.12
uv run --no-sync python benchmarks/audit_matched_training.py /tmp/matched.json
uv run --no-sync python benchmarks/audit_failure_recovery.py /tmp/failure.json
uv run --no-sync python benchmarks/audit_abrupt_recovery.py /tmp/abrupt.json
uv run --no-sync pytest -q tests/test_index_build_memory.py tests/test_ivfpq_index.py tests/test_serve_backends.py tests/test_serve_reconstruct.py tests/test_department_search.py
```

The abrupt audit kills only its own child processes, at controlled publication
boundaries, inside temporary fixtures. It does not kill services or fill disk.
All scripts take an explicit JSON output path; choose a new path to preserve
older evidence. The surrounding temporary data is removed on completion.

For memory measurements, choose new absent work directories:

```
uv run --no-sync python benchmarks/index_build_memory.py --work-dir /tmp/pixelrag-flat-large --rows 262144
uv run --no-sync python benchmarks/index_build_memory.py --work-dir /tmp/pixelrag-pq-large --rows 262144 --pq-m 16
```

Read index_build_memory.md for original methodology and source design.
Results under results/2026-10-26 and results/2026-10-27 are historical recordings,
not measurements from your machine. Their embedded work paths are provenance,
not required inputs. The three audit scripts were ported from standalone local
experiments and rerun from this independent checkout on 2026-10-28. Callback
bindings and formatting were cleaned up; the fixture scope is unchanged.

## What evidence supports

- At 262,144 input vectors: median peak RSS 960.80 -> 461.86 MiB (IVFFlat),
  692.02 -> 82.39 MiB (IVFPQ), with metadata alignment checks.
- With shared training input: identical top-10 IDs/scores on the 32,768-row
  fixture for IVFFlat/IVFPQ at two probing settings. This control deliberately
  overrides sampling and is not a performance benchmark.
- Four handled publication-stage errors: no partial destination; staging
  cleaned; clean retries succeed.
- SIGKILL just before/after rename: absent/readable destination respectively,
  but orphan staging survives. Rebuild does not resume or reclaim the orphan.

## What evidence does not support

No production speedup, semantic QA improvement, representative recall, Linux
validation, constant total RAM, power-loss durability, concurrent builders,
or general recovery guarantee is established. Synthetic PQ neighbor overlap
was poor. Largest compressed shard and final index still consume memory.
No browser, embedding model or paid API is required for these experiments.

Maintainer decisions remain: acceptable input/shard contract, representative
vector corpus, sampling policy, dedup costs, and ownership/durability of output.
