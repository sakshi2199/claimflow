"""Sentence-aware chunking with optional overlap.

Sentences are packed into a chunk until adding one more would exceed `chunk_size` characters. The next
chunk then starts with the trailing sentences of the previous one (up to `overlap` characters) so that
an idea split across a boundary appears whole in at least one chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.rag.documents import PolicyDocument

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<policy_id>::<index>"
    policy_id: str
    index: int
    text: str  # what is shown to the LLM
    embed_text: str  # what is embedded (text, optionally prefixed with the policy title)


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_BOUNDARY.split(" ".join(text.split())) if s]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in split_sentences(text):
        if current and length + len(sentence) + 1 > chunk_size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for previous in reversed(current):
                if carried + len(previous) + 1 > overlap:
                    break
                carry.insert(0, previous)
                carried += len(previous) + 1
            current, length = carry, carried
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_policy(document: PolicyDocument, chunk_size: int, overlap: int, include_title: bool) -> list[Chunk]:
    chunks = []
    for index, text in enumerate(chunk_text(document.text, chunk_size, overlap)):
        embed_text = f"{document.title}. {text}" if include_title else text
        chunks.append(Chunk(f"{document.policy_id}::{index}", document.policy_id, index, text, embed_text))
    return chunks
