# Proposed representative-vector evaluation

Status: protocol only; no representative fixture acquired or evaluated. This is
follow-up evidence for #179, not an approved integration policy.

## Input contract

Prefer an upstream-exported bounded sample of original precomputed vectors,
with permission to redistribute and a manifest recording source snapshot,
embedding-model revision, normalization, dimensions, row counts and SHA-256.
Use shard_*.npz files with embeddings (N,D) and aligned article_ids (int64),
tile_indices, chunk_indices, y_offsets and tile_heights (int32). Preserve source
order and duplicate occurrences; do not pre-deduplicate the input to conceal
builder differences. Record duplicate-key collisions separately: this proposal
preserves the legacy packed-key semantics, not a collision fix.

Target an initial <=256 MiB uncompressed vector fixture, with <=16 MiB vector
payload per shard. Choose row count from the real dimension (4*N*D bytes),
leaving headroom for metadata, training, the baseline copy and the final index.
These are proposed experiment limits, not production guarantees. Stratify by
source/article before sampling tiles; record strata and weights. A sample of
Wikipedia alone cannot establish behavior for PDFs or news.

Use held-out query embeddings from the same model revision, ideally real query
texts with permissible provenance. Keep queries separate from training samples.
If only held-out document vectors are available, label the task document-neighbor
search; do not describe it as question answering.

The public pixelrag-faiss-indexes inventory inspected on 2026-10-29 exposes
finished indexes and metadata, not original embedding shards. Reconstructing
vectors from a compressed index is a different experiment and must not silently
substitute for raw vectors. No multi-gigabyte download is required by this plan.

## Comparison matrix

1. Verify finite vectors, dimensions, norms and metadata alignment; record input
   hashes, duplicate counts and source/shard order. Fail on changed input.
2. Run the legacy builder and prototype with one shared training matrix first.
   Compare retained metadata exactly, result IDs and scores with declared numeric
   tolerance. This isolates addition/order correctness from sampling.
3. Run each builder with its own intended sampler, at five fixed recorded seeds,
   with equal training counts, nlist, nprobe, PQ settings and CPU threads. Report
   IVFFlat and IVFPQ separately; do not average away a failing configuration.
4. Construct an exact inner-product reference over the same retained vectors.
   Report recall@10 and @100 per query, mean, quantiles and paired differences
   between builders for each seed and stratum. Treat boundary ties explicitly.
   ANN agreement is not semantic relevance; QA accuracy requires labeled answers.
5. Measure peak RSS, wall time, output bytes and temporary-disk peak in fresh
   builder processes, separate from the exact reference and query evaluation.
   Alternate build order for three repetitions. Report hardware and caches.

## Review gates

Metadata/order errors block integration. Quality and resource tolerances require
maintainer agreement before treating results as a pass; no arbitrary acceptable
recall loss is assumed. Publish raw per-query/seed metrics even when worse.
A successful bounded experiment does not resolve whole-corpus shard limits,
concurrent writers, input snapshots, orphan ownership or power-loss durability.

Next action: obtain a small permissible raw-vector/query fixture, then implement
this protocol as an opt-in evaluator. The existing synthetic audits remain useful
regressions but cannot complete this work unit on their own.
