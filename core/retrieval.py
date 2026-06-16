# Nothing outside this file knows ChromaDB exists
# If we switch to Pinecone tomorrow — only this file changes

import time
import chromadb
from dataclasses import dataclass
from config import COLLECTION_PREFIX, MAX_RETRIEVAL_RESULTS
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

    def __init__(self):
        """
        Initialize ChromaDB with persistent storage.
        Data saved to ./chroma_db folder - survives restarts.
        """
        self._client = chromadb.PersistentClient(path="./chroma_db")
        self._collections: dict[str, object] = {}
        logger.info("repository_initialized", storage="persistent", path="./chroma_db")

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

        Args:
            doc_id:    unique identifier for this document
            chunks:    list of text chunks
            doc_name:  original filename for display
            pages:     page count, persisted for session restore
            file_type: file extension, persisted for session restore

        Returns:
            number of chunks stored
        """
        with Timer() as t:
            try:
                collection_name = f"{COLLECTION_PREFIX}{doc_id}"

                # This ensures clean state on re-upload
                try:
                    self._client.delete_collection(collection_name)
                    logger.info("existing_collection_deleted", name=collection_name)
                except Exception:
                    pass  # collection didn't exist — that's fine

                collection = self._client.create_collection(
                    name=collection_name,
                    metadata={
                        "doc_name": doc_name,
                        "hnsw:space": "cosine",
                        "pages": str(pages) if pages else "0",
                        "file_type": file_type if file_type else "unknown",
                        "created_at": str(time.time()),
                    },
                )

                for i, chunk in enumerate(chunks):
                    collection.add(
                        documents=[chunk],
                        ids=[f"chunk_{i}"],
                        metadatas=[
                            {
                                "doc_name": doc_name,
                                "chunk_index": i,
                                "chunk_length": len(chunk),
                            }
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

                    raw = collection.query(
                        query_texts=[question],
                        n_results=min(n_results, len(collection.get()["ids"])),
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
