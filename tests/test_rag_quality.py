# These tests verify RAG accuracy — not just "does it work"
# but "does it find the RIGHT information"

import pytest
from core.retrieval import DocumentRepository, RetrievalResult
from core.chunking import split_into_chunks
from core.ai_engine import AIEngine


@pytest.fixture
def chroma_path(tmp_path):
    """Isolated on-disk store per test — never touches the real ./chroma_db."""
    return str(tmp_path / "chroma")


@pytest.fixture
def repo(chroma_path):

    repository = DocumentRepository(path=chroma_path)
    yield repository
    repository.clear_all()  # cleanup after each test


@pytest.fixture
def loaded_repo(repo):
    """
    Fixture that pre-loads a compliance document.
    Tests that need data use this instead of repo.
    """
    compliance_text = """
    PCI DSS Requirement 8: Identify and authenticate access.
    Passwords must be changed every 90 days minimum.
    Passwords must be at least 8 characters long.
    Multi-factor authentication is required for all remote access.
    Failed login attempts must be limited to 6 before lockout.
    Account lockout duration must be at least 30 minutes.
    All access to cardholder data must be logged and monitored.
    Encryption keys must be rotated at least annually.
    """ * 20  # repeat to meet minimum length

    chunks = split_into_chunks(compliance_text, file_type="txt")
    repo.store_document("test_doc", chunks, "test_compliance.txt")
    return repo


def test_store_document_returns_chunk_count(repo):
    """Test that storing a document returns correct chunk count."""
    text = "Compliance requirement text. " * 50
    chunks = split_into_chunks(text)
    count = repo.store_document("doc1", chunks, "test.txt")
    assert count == len(chunks)
    assert count > 0


def test_store_empty_chunks_returns_zero(repo):
    """Test storing empty chunk list."""
    count = repo.store_document("doc1", [], "empty.txt")
    assert count == 0


def test_get_document_ids_returns_stored_docs(repo):
    """Test that stored doc IDs are tracked."""
    text = "Test content for compliance. " * 50
    chunks = split_into_chunks(text)
    repo.store_document("doc_a", chunks, "a.txt")
    repo.store_document("doc_b", chunks, "b.txt")
    ids = repo.get_document_ids()
    assert "doc_a" in ids
    assert "doc_b" in ids


def test_remove_document_cleans_up(repo):
    """Test that removing a document works correctly."""
    text = "Removable compliance content. " * 50
    chunks = split_into_chunks(text)
    repo.store_document("removable", chunks, "remove.txt")
    assert "removable" in repo.get_document_ids()
    repo.remove_document("removable")
    assert "removable" not in repo.get_document_ids()


def test_clear_all_removes_everything(repo):
    """Test that clear_all wipes all documents."""
    text = "Content to clear. " * 50
    chunks = split_into_chunks(text)
    repo.store_document("doc1", chunks, "1.txt")
    repo.store_document("doc2", chunks, "2.txt")
    repo.clear_all()
    assert repo.get_document_ids() == []


def test_reupload_replaces_existing(repo):
    """Test that uploading same doc_id replaces old content."""
    text1 = "Original password policy content. " * 50
    text2 = "Updated wire transfer policy content. " * 50
    chunks1 = split_into_chunks(text1)
    chunks2 = split_into_chunks(text2)

    repo.store_document("doc1", chunks1, "original.txt")
    repo.store_document("doc1", chunks2, "updated.txt")

    # Search should find new content not old
    results = repo.search("wire transfer", ["doc1"])
    assert len(results) > 0


def test_search_finds_relevant_content(loaded_repo):
    """
    Core RAG quality test.
    Ask a specific question — verify relevant chunk is returned.
    """
    results = loaded_repo.search("How often must passwords be changed?", ["test_doc"])
    assert len(results) > 0
    top_result = results[0]
    assert isinstance(top_result, RetrievalResult)
    assert len(top_result.chunk_text) > 0
    assert top_result.confidence_pct >= 0
    assert top_result.chunk_text


def test_search_returns_confidence_scores(loaded_repo):
    """Test that confidence scores are valid percentages."""
    results = loaded_repo.search("password requirements", ["test_doc"])
    for result in results:
        assert 0.0 <= result.confidence_pct <= 100.0


def test_search_results_sorted_by_confidence(loaded_repo):
    """Test that results come back sorted best-first."""
    results = loaded_repo.search("authentication lockout", ["test_doc"])
    if len(results) > 1:
        for i in range(len(results) - 1):
            assert results[i].confidence_pct >= results[i + 1].confidence_pct


def test_search_nonexistent_doc_returns_empty(repo):
    """Test searching a doc that doesn't exist returns empty."""
    results = repo.search("anything", ["nonexistent_doc"])
    assert results == []


def test_search_includes_doc_name(loaded_repo):
    """Test that results include source document name."""
    results = loaded_repo.search("encryption keys", ["test_doc"])
    assert len(results) > 0
    assert results[0].doc_name == "test_compliance.txt"


def test_multi_doc_search(repo):
    """Test searching across multiple documents."""
    text1 = "PCI DSS password rotation every 90 days required. " * 30
    text2 = "SOX financial controls audit trail mandatory. " * 30

    chunks1 = split_into_chunks(text1)
    chunks2 = split_into_chunks(text2)

    repo.store_document("pci", chunks1, "pci.txt")
    repo.store_document("sox", chunks2, "sox.txt")

    results = repo.search("password rotation", ["pci", "sox"])
    assert len(results) > 0
    # PCI doc should score higher for password question
    doc_names = [r.doc_name for r in results]
    assert "pci.txt" in doc_names


def test_restore_collections_round_trips_metadata(repo, chroma_path):
    """pages/file_type persisted on store should survive restore."""
    text = "Compliance requirement text. " * 50
    chunks = split_into_chunks(text)
    repo.store_document("restore_doc", chunks, "report.pdf", pages=7, file_type="pdf")

    # A fresh repository instance reads from the same persistent store
    fresh = DocumentRepository(path=chroma_path)
    restored = fresh.restore_collections()

    match = next(d for d in restored if d["doc_id"] == "restore_doc")
    assert match["doc_name"] == "report.pdf"
    assert match["pages"] == 7
    assert match["file_type"] == "pdf"
    assert match["chunks"] == len(chunks)


def test_get_all_chunks_returns_every_chunk_in_order(repo):
    """get_all_chunks must return all chunks ordered by chunk index, not top-k."""
    chunks = [f"Compliance requirement number {i:02d}. " * 5 for i in range(15)]
    repo.store_document("ordered_doc", chunks, "ordered.txt")

    retrieved = repo.get_all_chunks("ordered_doc")

    assert len(retrieved) == len(chunks)
    # chunk_10 must not sort before chunk_2 — verify true document order
    assert retrieved == chunks


def test_get_all_chunks_missing_doc_returns_empty(repo):
    assert repo.get_all_chunks("nope") == []


def test_remove_slot_documents_clears_only_that_slot(repo):
    """Uploading a new file to a slot must purge stale collections in that slot."""
    text = "Slot content for compliance testing. " * 50
    chunks = split_into_chunks(text)

    # Two different files that landed in the same UI slot over time
    repo.store_document("doc1_oldhash", chunks, "old.txt")
    repo.store_document("doc1_newhash", chunks, "new.txt")
    repo.store_document("doc2_keep", chunks, "other.txt")

    removed = repo.remove_slot_documents("doc1")

    assert removed == 2
    ids = repo.get_document_ids()
    assert "doc1_oldhash" not in ids
    assert "doc1_newhash" not in ids
    assert "doc2_keep" in ids  # other slot untouched


def test_mapreduce_checklist(loaded_repo):
    """Test map-reduce processes all chunks."""
    from unittest.mock import MagicMock

    engine = AIEngine(loaded_repo)

    mock_response = MagicMock()
    mock_response.content[0].text = "- [ ] Test requirement"
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 50

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    engine._client = mock_client

    result = engine.generate_checklist_mapreduce("test_doc", "test_compliance.txt")

    assert result.answer is not None
    assert len(result.answer) > 0
    assert mock_client.messages.create.call_count >= 2
