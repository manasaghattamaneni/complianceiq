import pytest
import io
from unittest.mock import MagicMock, patch
from core.ingestion import extract_text, _read_txt, _read_csv


def test_read_txt_returns_text():
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_file.read.return_value = (
        b"This is a test document with compliance content. " * 5
    )
    text, pages, file_type = extract_text(file=mock_file)
    assert "test document" in text
    assert file_type == "txt"
    assert pages >= 1


def test_read_txt_handles_latin_encoding():
    mock_file = MagicMock()
    mock_file.name = "test.txt"
    mock_file.read.return_value = ("Héllo Wörld " * 20).encode("latin-1")
    text, pages, file_type = extract_text(file=mock_file)
    assert len(text) > 0


def test_read_csv_returns_structured_text():
    """Test CSV extraction includes column names."""
    csv_content = b"name,role,department\nManasa,Engineer,Tech\nJohn,Manager,Finance"
    mock_file = MagicMock()
    mock_file.name = "test.csv"

    # io.BytesIO wraps bytes as a file-like object
    # pandas read_csv needs a file-like object, not a MagicMock
    with patch("core.ingestion.pd.read_csv") as mock_csv:
        import pandas as pd

        mock_csv.return_value = pd.DataFrame(
            {
                "name": ["Manasa", "John"],
                "role": ["Engineer", "Manager"],
                "department": ["Tech", "Finance"],
            }
        )
        text, pages, file_type = extract_text(file=mock_file)
        assert "Columns" in text
        assert file_type == "csv"


def test_unsupported_file_type_raises_error():
    """Test that unsupported file types raise ValueError."""
    mock_file = MagicMock()
    mock_file.name = "malware.exe"
    mock_file.read.return_value = b"fake content " * 20
    with pytest.raises(ValueError):
        extract_text(file=mock_file)


def test_empty_content_raises_error():
    """Test that empty documents raise ValueError."""
    mock_file = MagicMock()
    mock_file.name = "empty.txt"
    mock_file.read.return_value = b"   "
    with pytest.raises(ValueError):
        extract_text(file=mock_file)


def test_short_content_raises_error():
    """Test that documents with too little text raise ValueError."""
    mock_file = MagicMock()
    mock_file.name = "tiny.txt"
    mock_file.read.return_value = b"Too short"
    with pytest.raises(ValueError):
        extract_text(file=mock_file)
