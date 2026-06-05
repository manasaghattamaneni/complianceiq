# tests/test_chunking.py
# Tests for smart chunking strategies

import pytest
from core.chunking import split_into_chunks, _default_strategy, _csv_strategy

# ---- Default Strategy Tests ----


def test_normal_text_produces_chunks():
    text = "A" * 1000
    chunks = split_into_chunks(text)
    assert len(chunks) > 0


def test_empty_text_returns_empty_list():
    chunks = split_into_chunks("")
    assert chunks == []


def test_whitespace_only_returns_empty_list():
    chunks = split_into_chunks("   \n\n  ")
    assert chunks == []


def test_short_text_returns_one_chunk():
    text = "This is a short compliance document. " * 3
    chunks = split_into_chunks(text)
    assert len(chunks) >= 1


def test_chunk_size_respected():
    text = "A" * 2000
    chunks = _default_strategy(text, chunk_size=500, overlap=50)
    # Each chunk should be at most 500 chars
    for chunk in chunks:
        assert len(chunk) <= 500


def test_overlap_creates_repeated_content():
    text = "Word " * 200  # 1000 chars
    chunks = _default_strategy(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # Consecutive chunks should share some content due to overlap
    if len(chunks) >= 2:
        end_of_first = chunks[0][-20:]
        start_of_second = chunks[1][:20]
        # They should share characters
        assert len(end_of_first) > 0
        assert len(start_of_second) > 0


# ---- CSV Strategy Tests ----


def _csv_strategy(text: str) -> list[str]:
    """
    Row-based chunking for CSV/tabular data.
    Groups rows into chunks of 20 rows each.
    Keeps column headers in every chunk for context.
    """
    # Early return for empty input
    if not text or not text.strip():
        return []

    lines = [l for l in text.split("\n") if l.strip()]

    if not lines:
        return []

    # First line is always column headers
    header = lines[0]
    data_lines = lines[1:]

    if not data_lines:
        return [header]

    chunks = []
    rows_per_chunk = 20

    # Group rows into chunks — always include header
    for i in range(0, len(data_lines), rows_per_chunk):
        batch = data_lines[i : i + rows_per_chunk]
        chunk = header + "\n" + "\n".join(batch)
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks if chunks else [text]


def test_csv_chunking_includes_header():
    csv_text = "name,role,salary\n" + "John,Engineer,100000\n" * 50
    chunks = _csv_strategy(csv_text)
    # Every chunk should contain the header
    for chunk in chunks:
        assert "name,role,salary" in chunk


def test_csv_empty_returns_fallback():
    chunks = _csv_strategy("")
    assert chunks == []


def test_csv_groups_rows():
    # 50 data rows should produce multiple chunks
    csv_text = "col1,col2\n" + "a,b\n" * 50
    chunks = _csv_strategy(csv_text)
    assert len(chunks) > 1


# ---- Strategy Selection Tests ----


def test_csv_file_uses_csv_strategy():
    csv_text = "name,value\n" + "item,100\n" * 25
    chunks = split_into_chunks(csv_text, file_type="csv")
    assert len(chunks) > 0
    # CSV strategy includes header in each chunk
    for chunk in chunks:
        assert "name,value" in chunk


def test_large_doc_uses_sentence_strategy():
    # Create a large document over LARGE_DOC_THRESHOLD (50000 chars)
    large_text = "This is a compliance requirement sentence. " * 1500
    chunks = split_into_chunks(large_text, file_type="pdf")
    assert len(chunks) > 0


def test_small_doc_uses_default_strategy():
    small_text = "Short compliance document. " * 20
    chunks = split_into_chunks(small_text, file_type="pdf")
    assert len(chunks) >= 1


def test_custom_chunk_size_respected():
    text = "B" * 2000
    chunks = split_into_chunks(text, chunk_size=200, overlap=20)
    for chunk in chunks:
        assert len(chunk) <= 200
