"""Embedding wrapper around ``sentence-transformers``.

Embeddings are L2-normalized at generation time so the HNSW index can use
inner-product distance as a cosine equivalent without a second normalization.
"""

from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from src.config import get_settings


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> SentenceTransformer:
    return cast(SentenceTransformer, SentenceTransformer(model_name))


def get_model() -> SentenceTransformer:
    return _load_model(get_settings().embedding_model)


def embed(text: str) -> NDArray[np.float32]:
    """Return a unit-norm float32 embedding for ``text`` as a 1-D array."""

    vector = get_model().encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return cast(NDArray[np.float32], np.asarray(vector, dtype=np.float32))
