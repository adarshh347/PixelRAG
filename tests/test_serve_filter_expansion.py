"""Post-filter candidate-window regression; no model or external service."""

from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixelrag_serve import api


class RankedBackend:
    dimension = 4

    def __init__(self):
        self.calls = []
        self.options = []
        self.size = 22
        self.fail_call = None

    def set_nprobe(self, n):
        raise AssertionError("request must not mutate backend settings")

    def reset_nprobe(self):
        raise AssertionError("request must not mutate backend settings")

    def raw_search(self, query_vectors, k, **kwargs):
        self.calls.append((query_vectors.copy(), k))
        self.options.append(kwargs)
        if len(self.calls) == self.fail_call:
            raise RuntimeError("backend unavailable")
        return [
            [
                {
                    "score": 1.0 - i / 100,
                    "vector_id": i,
                    "article_id": i,
                    "tile_index": 0,
                    "chunk_index": 0,
                    "y_offset": 0,
                    "tile_height": 100,
                }
                for i in range(self.size)
                if kwargs.get("article_ids") is None or i in kwargs["article_ids"]
            ][:k]
            for _ in query_vectors
        ]


@pytest.fixture
def ranked_client(monkeypatch, tmp_path):
    backend = RankedBackend()
    monkeypatch.setattr(
        api,
        "_state",
        {
            "backend": backend,
            "dimension": backend.dimension,
            "articles": [f"Portal:Meta_{i}" for i in range(20)]
            + ["Eligible_first", "Eligible_second"],
            "tiles_dir": str(tmp_path),
        },
    )
    encode = Mock(side_effect=AssertionError("must use supplied embeddings"))
    monkeypatch.setattr(api, "_encode_queries", encode)
    api._article_pages.cache_clear()
    with TestClient(api.app) as client:
        yield client, backend, encode
    api._article_pages.cache_clear()


def test_articles_beyond_initial_candidate_window(ranked_client):
    client, backend, encode = ranked_client
    response = client.post(
        "/search",
        json={
            "queries": [{"embedding": [1.0, 0.0, 0.0, 0.0]}],
            "n_docs": 2,
            "articles_only": True,
        },
    )
    assert response.status_code == 200, response.text
    assert backend.calls[0][1] == 20
    encode.assert_not_called()
    assert [hit["vector_id"] for hit in response.json()["results"][0]["hits"]] == [
        20,
        21,
    ]
    for vectors, _ in backend.calls:
        np.testing.assert_array_equal(vectors, [[1.0, 0.0, 0.0, 0.0]])


def _search(client, **overrides):
    body = {
        "queries": [{"embedding": [1.0, 0.0, 0.0, 0.0]}],
        "n_docs": 2,
        "articles_only": True,
    }
    body.update(overrides)
    response = client.post("/search", json=body)
    assert response.status_code == 200, response.text
    return response.json()["results"]


@pytest.mark.parametrize("articles_only", [False, True])
def test_first_batch_suffices(ranked_client, articles_only):
    client, backend, _ = ranked_client
    api._state["articles"] = [f"Article_{i}" for i in range(22)]
    assert len(_search(client, articles_only=articles_only)[0]["hits"]) == 2
    assert [k for _, k in backend.calls] == [20 if articles_only else 2]


@pytest.mark.parametrize(
    "size, expected_calls", [(0, [20]), (7, [20]), (22, [20, 40]), (200, [20, 40])]
)
def test_no_matches_stops_at_shortage_or_budget(ranked_client, size, expected_calls):
    client, backend, _ = ranked_client
    backend.size = size
    api._state["articles"] = [f"Portal:Meta_{i}" for i in range(size)]
    assert _search(client)[0]["hits"] == []
    assert [k for _, k in backend.calls] == expected_calls


def test_combined_filters_and_department(ranked_client):
    client, backend, _ = ranked_client
    backend.size = 24
    api._state["articles"] += ["Eligible_third", "Eligible_fourth"]
    api._state["dept_to_aids"] = {"eng": list(range(22))}
    results = _search(client, min_tile_height=100, department="eng")
    assert [h["vector_id"] for h in results[0]["hits"]] == [20, 21]
    assert len(backend.calls) == 2
    assert all(o["filter_cache_key"] == "dept:eng" for o in backend.options)
    assert all(o["min_tile_height"] == 100 for o in backend.options)
    backend.calls.clear()
    assert _search(client, min_tile_height=101, department="eng")[0]["hits"] == []


def test_budget_is_shared_across_queries(ranked_client, monkeypatch):
    client, backend, _ = ranked_client
    monkeypatch.setattr(api, "_MAX_EXPANSION_CANDIDATES", 60)
    queries = [{"embedding": [1.0, 0.0, 0.0, 0.0]}] * 3
    results = _search(client, queries=queries)
    assert all(r["hits"] == [] for r in results)
    # Remaining allowance cannot buy a larger window for all three queries.
    assert [k for _, k in backend.calls] == [20]


def test_multiple_queries_expand_and_render_only_final_hits(ranked_client, monkeypatch):
    client, backend, encode = ranked_client
    ondemand = Mock()
    ondemand.cached_chunk_path.return_value = None
    api._state["ondemand"] = ondemand
    render = Mock(return_value="image")
    monkeypatch.setattr(api, "_ondemand_chunk_b64", render)
    monkeypatch.setattr(api, "_MAX_ONDEMAND_RENDERS", 3)
    queries = [{"embedding": [1.0, 0.0, 0.0, 0.0]}] * 2
    results = _search(client, queries=queries, include_images=True)
    assert [[h["vector_id"] for h in r["hits"]] for r in results] == [
        [20, 21],
        [20, 21],
    ]
    assert [k for _, k in backend.calls] == [20, 40]
    assert render.call_count == 3
    assert [call.args[0] for call in render.call_args_list] == [20, 21, 20]
    assert results[1]["hits"][1]["image_base64"] is None
    encode.assert_not_called()


def test_backend_error_does_not_leak_settings_or_hydrate(ranked_client, monkeypatch):
    client, backend, _ = ranked_client
    resolve_path = api._resolve_path
    hydrate = Mock(side_effect=AssertionError("must select hits before hydration"))
    monkeypatch.setattr(api, "_resolve_path", hydrate)
    backend.fail_call = 2
    with pytest.raises(RuntimeError, match="backend unavailable"):
        _search(client, nprobe=7, include_images=True)
    hydrate.assert_not_called()
    assert all(o["nprobe"] == 7 for o in backend.options)
    backend.fail_call = None
    monkeypatch.setattr(api, "_resolve_path", resolve_path)
    _search(client)
    assert backend.options[-1].get("nprobe") is None


def test_initial_partial_hit_is_not_hydrated_again_during_expansion(
    ranked_client, monkeypatch
):
    client, backend, _ = ranked_client
    api._state["articles"][0] = "Initially_eligible"
    ondemand = Mock()
    ondemand.cached_chunk_path.return_value = None
    api._state["ondemand"] = ondemand
    render = Mock(return_value="image")
    monkeypatch.setattr(api, "_ondemand_chunk_b64", render)
    original = backend.raw_search

    def search_before_hydration(*args, **kwargs):
        render.assert_not_called()
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "raw_search", search_before_hydration)
    hits = _search(client, include_images=True)[0]["hits"]
    assert [h["vector_id"] for h in hits] == [0, 20]
    assert [k for _, k in backend.calls] == [20, 40]
    assert [call.args[0] for call in render.call_args_list] == [0, 20]


def test_duplicate_candidates_do_not_duplicate_final_hits(ranked_client, monkeypatch):
    client, backend, _ = ranked_client
    original = backend.raw_search

    def duplicates(vectors, k, **kwargs):
        return [
            [*row[:21], row[20], *row[21:]] if len(row) > 20 else row
            for row in original(vectors, k, **kwargs)
        ]

    monkeypatch.setattr(backend, "raw_search", duplicates)
    hits = _search(client)[0]["hits"]
    assert [h["vector_id"] for h in hits] == [20, 21]


def test_only_pending_queries_expand_and_second_round_is_bounded(
    ranked_client, monkeypatch
):
    client, backend, _ = ranked_client
    backend.size = 100
    api._state["articles"] = [f"Portal:Meta_{i}" for i in range(100)]
    for i in (20, 21, 40, 41):
        api._state["articles"][i] = f"Article_{i}"
    original = backend.raw_search

    def ranked(vectors, k, **kwargs):
        rows = original(vectors, k, **kwargs)
        for vector, row in zip(vectors, rows, strict=True):
            if vector[1]:
                # Second query's first eligible hits are ranks 41 and 42.
                for hit in row:
                    if hit["article_id"] in (20, 21):
                        hit["article_id"] = 0
        return rows

    monkeypatch.setattr(backend, "raw_search", ranked)
    results = _search(
        client,
        queries=[
            {"embedding": [1.0, 0.0, 0.0, 0.0]},
            {"embedding": [0.0, 1.0, 0.0, 0.0]},
            {"embedding": [1.0, 0.0, 0.0, 0.0]},
        ],
    )
    assert [[h["vector_id"] for h in r["hits"]] for r in results] == [
        [20, 21],
        [40, 41],
        [20, 21],
    ]
    # Initial 60 slots; 180 extra slots shared. 120 buys round one,
    # leaving 60 for the only unfinished query's second expansion.
    assert [(len(v), k) for v, k in backend.calls] == [(3, 20), (3, 40), (1, 60)]
    np.testing.assert_array_equal(backend.calls[-1][0], [[0, 1, 0, 0]])


def test_height_only_expands(ranked_client, monkeypatch):
    client, backend, _ = ranked_client
    original = backend.raw_search

    def short_tiles(vectors, k, **kwargs):
        rows = original(vectors, k, **kwargs)
        for row in rows:
            for hit in row:
                if hit["vector_id"] < 10:
                    hit["tile_height"] = 1
        return rows

    monkeypatch.setattr(backend, "raw_search", short_tiles)
    hits = _search(client, articles_only=False, min_tile_height=100)[0]["hits"]
    assert [h["vector_id"] for h in hits] == [10, 11]
    assert [k for _, k in backend.calls] == [10, 20]


def test_round_cap_and_completed_queries(ranked_client, monkeypatch):
    client, backend, _ = ranked_client
    backend.size = 500
    api._state["articles"] = [f"Portal:Meta_{i}" for i in range(500)]
    api._state["articles"][0:2] = ["First", "Second"]
    original = backend.raw_search

    def ranked(vectors, k, **kwargs):
        rows = original(vectors, k, **kwargs)
        for vector, row in zip(vectors, rows, strict=True):
            if vector[1]:
                for hit in row:
                    hit["article_id"] = 2
        return rows

    monkeypatch.setattr(backend, "raw_search", ranked)
    queries = [{"embedding": [1, 0, 0, 0]}] * 3 + [{"embedding": [0, 1, 0, 0]}]
    results = _search(client, queries=queries)
    assert [len(r["hits"]) for r in results] == [2, 2, 2, 0]
    assert [(len(v), k) for v, k in backend.calls] == [(4, 20), (1, 40), (1, 80)]
    # 120 candidate slots still remain, enough for a larger window, but
    # two retries is the separate call limit. Completed rows never retry.


def test_final_window_controls_ranking(ranked_client, monkeypatch):
    client, backend, _ = ranked_client
    api._state["articles"][0] = "Initially_eligible"
    original = backend.raw_search

    def reranked(vectors, k, **kwargs):
        rows = original(vectors, k, **kwargs)
        if k > 20:
            # Simulate an approximate backend changing order with larger k.
            for row in rows:
                row.insert(0, row.pop(21))
                row[0]["score"] = 2.0
        return rows

    monkeypatch.setattr(backend, "raw_search", reranked)
    hits = _search(client)[0]["hits"]
    assert [h["vector_id"] for h in hits] == [21, 0]
    assert [h["score"] for h in hits] == [2.0, 1.0]


@pytest.mark.parametrize("first_eligible, expected", [(39, [39]), (40, [])])
def test_cap_can_return_partial_or_empty_results(
    ranked_client, first_eligible, expected
):
    client, backend, _ = ranked_client
    backend.size = 80
    api._state["articles"] = [
        f"Portal:Meta_{i}" if i < first_eligible else f"Article_{i}" for i in range(80)
    ]
    hits = _search(client)[0]["hits"]
    assert [h["vector_id"] for h in hits] == expected
    assert [k for _, k in backend.calls] == [20, 40]
