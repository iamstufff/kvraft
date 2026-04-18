import numpy as np
import pytest
from numpy.typing import NDArray

from src.cache.index import SemanticIndex


def _unit(values: list[float]) -> NDArray[np.float32]:
    array = np.array(values, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def small_index() -> SemanticIndex:
    return SemanticIndex(dim=4, initial_capacity=16)


def test_size_reports_inserted_count(small_index: SemanticIndex) -> None:
    assert small_index.size == 0
    small_index.add(_unit([1.0, 0.0, 0.0, 0.0]), id_=1)
    assert small_index.size == 1
    small_index.add(_unit([0.0, 1.0, 0.0, 0.0]), id_=2)
    assert small_index.size == 2


def test_exact_match_returns_high_similarity(small_index: SemanticIndex) -> None:
    vec = _unit([1.0, 0.0, 0.0, 0.0])
    small_index.add(vec, id_=42)

    matches = small_index.search(vec, k=1, threshold=0.9)

    assert len(matches) == 1
    assert matches[0].id == 42
    assert matches[0].similarity == pytest.approx(1.0, abs=1e-5)


def test_search_filters_below_threshold(small_index: SemanticIndex) -> None:
    near = _unit([1.0, 0.0, 0.0, 0.0])
    orthogonal = _unit([0.0, 1.0, 0.0, 0.0])
    close_to_near = _unit([0.95, 0.3, 0.05, 0.02])
    small_index.add(near, id_=1)
    small_index.add(orthogonal, id_=2)
    small_index.add(close_to_near, id_=3)

    matches = small_index.search(near, k=3, threshold=0.9)
    ids = {m.id for m in matches}

    assert 1 in ids
    assert 3 in ids
    assert 2 not in ids


def test_search_on_empty_index_returns_empty(small_index: SemanticIndex) -> None:
    result = small_index.search(_unit([1.0, 0.0, 0.0, 0.0]), k=5, threshold=0.5)
    assert result == []


def test_capacity_doubles_when_full() -> None:
    index = SemanticIndex(dim=4, initial_capacity=2)
    index.add(_unit([1.0, 0.0, 0.0, 0.0]), id_=1)
    index.add(_unit([0.0, 1.0, 0.0, 0.0]), id_=2)
    # Third insert must trigger a resize rather than raising.
    index.add(_unit([0.0, 0.0, 1.0, 0.0]), id_=3)
    assert index.size == 3


def test_search_returns_nothing_when_all_below_threshold(small_index: SemanticIndex) -> None:
    small_index.add(_unit([0.0, 1.0, 0.0, 0.0]), id_=1)
    matches = small_index.search(_unit([1.0, 0.0, 0.0, 0.0]), k=1, threshold=0.5)
    assert matches == []
