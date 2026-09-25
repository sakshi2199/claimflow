"""Embedding models. Chroma only stores the vectors; we compute them here so the embedder is explicit."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 (384-d) via Chroma's bundled ONNX runtime. The model (~80 MB) is downloaded on first use."""

    name = "all-MiniLM-L6-v2-onnx"

    def __init__(self) -> None:
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

            self._model = ONNXMiniLM_L6_V2()
        return [[float(x) for x in vector] for vector in self._model(texts)]


class HashingEmbedder:
    """Offline, deterministic bag-of-words embedding (feature hashing). Lexical, not semantic.

    Used by unit tests so they need no model download. Never used for reported results.
    """

    name = "hashing-384"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()  # stable across runs, unlike hash()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]


def get_embedder(kind: str) -> Embedder:
    if kind == "minilm":
        return MiniLMEmbedder()
    if kind == "hashing":
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {kind!r} (expected 'minilm' or 'hashing')")
