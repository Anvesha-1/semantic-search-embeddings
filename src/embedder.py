"""Member 2: turns text into embeddings (lists of 384 numbers)."""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts, batch_size=64, show_progress=False):
    """Convert a string or a list of strings into a float32 array.

    Output shape: (number_of_texts, 384).
    Each row is normalized to length 1, so cosine similarity
    equals the inner product.
    """
    if isinstance(texts, str):
        texts = [texts]
    model = get_model()
    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return embeddings.astype(np.float32)