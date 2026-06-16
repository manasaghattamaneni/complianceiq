# Nothing outside this file knows ChromaDB exists
# If we switch to Pinecone tomorrow — only this file changes

import time
import chromadb
from dataclasses import dataclass
from config import COLLECTION_PREFIX, MAX_RETRIEVAL_RESULTS, CHROMA_DB_PATH
from utils.logger import logger
from utils.metrics import Timer


@dataclass
class RetrievalResult:

    chunk_text: str
    confidence_pct: float
    doc_name: str
    chunk_index: int


class DocumentRepository:
    """
    Repository for storing and searching document chunks.
    Single Responsibility: ONLY manages ChromaDB operations.

    Usage:
        repo = DocumentRepository()
        repo.store_document("pci_dss", chunks, "pci_dss.pdf")
        results = repo.search("password requirements", ["pci_dss"])
    """

    def __init__(self, path: str = CHROMA_DB_PATH):
        """
        Initialize ChromaDB with persistent storage.
        Data is saved to ``path`` and survives restarts.

        ``path`` is injectable so tests can point at a temp directory
        instead of polluting the real ./chroma_db store.
        """
        self._client = chromadb.PersistentClient(path=path)
        self._collections: dict[str, object] = {}
        logger.info("repository_initialized", storage="persistent", path=path)

    def store_document(
        self,
        doc_id: str,
        chunks: list[str],
        doc_name: str,
        pages: int = 0,
        file_type: str = "unknown",
    ) -> int:
        """
        Store document chunks in ChromaDB.
        Deletes any existing collection with this exact doc_id
        before creating the new one — ensures clean re-upload.
        """
        with Timer() as t:
            try:
                collection_name = f"{COLLECTION_PREFIX}{doc_id}"

                try:
                    self._client.delete_collection(collection_name)
                    logger.info("existing_collection_deleted", name=collection_name)
                except Exception:
                    pass

                collection = self._client.create_collection(
                    name=collection_name,
                    metadata={
                        "doc_name": doc_name,
                        "hnsw:space": "cosine",
                        "pages": str(pages),
                        "file_type": file_type,
                        "created_at": str(time.time()),
                    },
                )

                # Single batched insert — embeds all chunks in one pass
                # instead of one (slow) embedding call per chunk.
                if chunks:
                    collection.add(
                        documents=list(chunks),
                        ids=[f"chunk_{i}" for i in range(len(chunks))],
                        metadatas=[
                            {
                                "doc_name": doc_name,
                                "chunk_index": i,
                                "chunk_length": len(chunk),
                            }
                            for i, chunk in enumerate(chunks)
                        ],
                    )

                self._collections[doc_id] = collection

                logger.info(
                    "document_stored",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    chunks_stored=len(chunks),
                    duration_ms=t.duration_ms,
                )

                return len(chunks)

            except Exception as e:
                logger.log_error("store_failed", e, doc_name=doc_name)
                raise ValueError(f"Failed to store document: {str(e)}")

    def search(
        self, question: str, doc_ids: list[str], n_results: int = None
    ) -> list[RetrievalResult]:
        """
        Search for relevant chunks across one or more documents.

        Args:
            question: user's question
            doc_ids:  list of document IDs to search
            n_results: how many results per document

        Returns:
            list of RetrievalResult sorted by confidence
        """
        if not n_results:
            n_results = MAX_RETRIEVAL_RESULTS

        with Timer() as t:
            try:
                all_results = []

                for doc_id in doc_ids:
                    collection = self._collections.get(doc_id)
                    if not collection:
                        logger.warning("collection_not_found", doc_id=doc_id)
                        continue

                    count = collection.count()
                    if count == 0:
                        continue

                    raw = collection.query(
                        query_texts=[question],
                        n_results=min(n_results, count),
                    )

                    chunks = raw["documents"][0]
                    distances = raw["distances"][0]
                    metadatas = raw["metadatas"][0]

                    for chunk, distance, metadata in zip(chunks, distances, metadatas):
                        # cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
                        # similarity = (1 - distance) * 100
                        confidence = round((1 - distance) * 100, 1)
                        confidence = max(0.0, min(100.0, confidence))

                        all_results.append(
                            RetrievalResult(
                                chunk_text=chunk,
                                confidence_pct=confidence,
                                doc_name=metadata.get("doc_name", doc_id),
                                chunk_index=metadata.get("chunk_index", 0),
                            )
                        )

                all_results.sort(key=lambda r: r.confidence_pct, reverse=True)

                logger.info(
                    "search_complete",
                    question_length=len(question),
                    docs_searched=len(doc_ids),
                    results_found=len(all_results),
                    top_confidence=all_results[0].confidence_pct if all_results else 0,
                    duration_ms=t.duration_ms,
                )

                return all_results

            except Exception as e:
                logger.log_error("search_failed", e)
                raise ValueError(f"Search failed: {str(e)}")

    def remove_document(self, doc_id: str) -> None:
        """
        Remove a document from the repository.
        Called when user uploads a new document.
        """
        try:
            collection_name = f"{COLLECTION_PREFIX}{doc_id}"
            self._client.delete_collection(collection_name)
            self._collections.pop(doc_id, None)
            logger.info("document_removed", doc_id=doc_id)
        except Exception:
            pass  # already gone — that's fine

    def get_document_ids(self) -> list[str]:
        """Return list of currently loaded document IDs."""
        return list(self._collections.keys())

    def clear_all(self) -> None:
        """
        Remove all documents — called on session reset.
        Ensures clean state between sessions.
        """
        for doc_id in list(self._collections.keys()):
            self.remove_document(doc_id)
        logger.info("repository_cleared")

    def restore_collections(self, limit: int = 2) -> list[dict]:
        """
        Reload the most recently created collections from disk.

        Collections are ranked by their stored ``created_at`` timestamp and
        capped at ``limit`` so a restart never surfaces arbitrary old
        documents. Returns metadata dicts the UI uses to repopulate its slots.
        """
        try:
            candidates = []

            for col in self._client.list_collections():
                name = col.name
                if not name.startswith(COLLECTION_PREFIX):
                    continue
                try:
                    doc_id = name[len(COLLECTION_PREFIX) :]
                    collection = self._client.get_collection(name)
                    metadata = collection.metadata or {}
                    candidates.append(
                        {
                            "doc_id": doc_id,
                            "doc_name": metadata.get("doc_name", doc_id),
                            "chunks": collection.count(),
                            "pages": int(metadata.get("pages", 0)),
                            "file_type": metadata.get("file_type", "unknown"),
                            "created_at": float(metadata.get("created_at", 0.0)),
                            "collection": collection,
                        }
                    )
                except Exception as e:
                    # One bad collection shouldn't abort the whole restore
                    logger.log_error("collection_restore_skipped", e)

            # Most recent first, then keep only what the UI can show
            candidates.sort(key=lambda c: c["created_at"], reverse=True)
            candidates = candidates[:limit]

            restored = []
            for c in candidates:
                self._collections[c["doc_id"]] = c["collection"]
                restored.append(
                    {
                        "doc_id": c["doc_id"],
                        "doc_name": c["doc_name"],
                        "chunks": c["chunks"],
                        "pages": c["pages"],
                        "file_type": c["file_type"],
                    }
                )
                logger.info(
                    "collection_restored",
                    doc_id=c["doc_id"],
                    doc_name=c["doc_name"],
                )

            logger.info("restore_complete", count=len(restored))
            return restored

        except Exception as e:
            logger.log_error("restore_failed", e)
            return []

    def get_all_chunks(self, doc_id: str) -> list[str]:
        """
        Return all chunks for a document.
        Used by map-reduce operations that need every chunk,
        not just top-k search results.
        Keeps ChromaDB encapsulated inside this file.
        """
        collection = self._collections.get(doc_id)
        if not collection:
            return []
        data = collection.get()
        ids = data["ids"]
        documents = data["documents"]

        # ChromaDB.get() does not guarantee insertion order, and ids sort
        # lexicographically (chunk_10 < chunk_2). Reorder by the numeric
        # suffix so map-reduce sees chunks in document order.
        def _chunk_index(chunk_id: str) -> int:
            try:
                return int(chunk_id.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                return 0

        ordered = sorted(zip(ids, documents), key=lambda pair: _chunk_index(pair[0]))
        return [doc for _, doc in ordered]

    def remove_slot_documents(self, slot_prefix: str) -> int:
        """
        Remove all collections belonging to a given upload slot
        (e.g. 'doc1' or 'doc2') before storing a new document there.
        Prevents orphaned collections when a different file is
        uploaded to the same UI slot — production use case only,
        not used by the generic repository tests.

        Returns count of collections removed.
        """
        removed = 0
        try:
            existing = self._client.list_collections()
            for col in existing:
                name_without_prefix = col.name[len(COLLECTION_PREFIX) :]
                if name_without_prefix.startswith(f"{slot_prefix}_"):
                    try:
                        self._client.delete_collection(col.name)
                        self._collections.pop(name_without_prefix, None)
                        removed += 1
                        logger.info(
                            "stale_slot_collection_deleted",
                            name=col.name,
                            slot=slot_prefix,
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.log_error("remove_slot_documents_failed", e)
        return removed
