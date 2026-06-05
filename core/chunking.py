# core/chunking.py
import re
from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    LARGE_DOC_THRESHOLD,
    LARGE_DOC_CHUNK_SIZE,
    LARGE_DOC_OVERLAP,
)
from utils.logger import logger


def _default_strategy(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Default chunking — splits by character count with overlap."""
    if overlap >= chunk_size:
        overlap = chunk_size // 4

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = max(end - overlap, start + 1)
    return chunks


def _sentence_aware_strategy(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Sentence-aware chunking using regex — O(n) not O(n²).
    Handles abbreviations like U.S., e.g., Section 8.2.1
    """
    sentence_pattern = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+")
    sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            if len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def _csv_strategy(text: str) -> list[str]:
    """
    Row-based chunking for CSV/tabular data.
    Keeps column headers in every chunk for context.
    """
    if not text or not text.strip():
        return []

    lines = [l for l in text.split("\n") if l.strip()]

    if not lines:
        return []

    header = lines[0]
    data_lines = lines[1:]

    if not data_lines:
        return [header]

    chunks = []
    rows_per_chunk = 20

    for i in range(0, len(data_lines), rows_per_chunk):
        batch = data_lines[i : i + rows_per_chunk]
        chunk = header + "\n" + "\n".join(batch)
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks if chunks else [text]


def split_into_chunks(
    text: str, file_type: str = "txt", chunk_size: int = None, overlap: int = None
) -> list[str]:
    """
    Strategy Pattern entry point.
    Picks the right chunking strategy based on file type
    and document size.
    """
    if not text or not text.strip():
        return []

    if chunk_size is None:
        chunk_size = (
            LARGE_DOC_CHUNK_SIZE
            if len(text) > LARGE_DOC_THRESHOLD
            else DEFAULT_CHUNK_SIZE
        )

    if overlap is None:
        overlap = (
            LARGE_DOC_OVERLAP if len(text) > LARGE_DOC_THRESHOLD else DEFAULT_OVERLAP
        )

    if file_type == "csv":
        chunks = _csv_strategy(text)
        strategy_used = "csv_row_based"
    elif len(text) > LARGE_DOC_THRESHOLD:
        chunks = _sentence_aware_strategy(text, chunk_size, overlap)
        strategy_used = "sentence_aware"
    else:
        chunks = _default_strategy(text, chunk_size, overlap)
        strategy_used = "default"

    chunks = [c for c in chunks if len(c.strip()) > 20]

    logger.info(
        "chunking_complete",
        strategy=strategy_used,
        file_type=file_type,
        total_chars=len(text),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    return chunks
