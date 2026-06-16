import anthropic
import streamlit as st
from dataclasses import dataclass
from config import AI_MODEL, MAX_TOKENS
from core.retrieval import DocumentRepository, RetrievalResult
from utils.logger import logger
from utils.metrics import Timer, QueryMetric


@dataclass
class AIResponse:
    """Structured response from Claude API."""

    answer: str
    token_count: int
    top_confidence: float
    sources: list[RetrievalResult]


class AIEngine:
    """
    Handles all Claude API interactions.
    Single Responsibility: ONLY calls Claude.

    Usage:
        engine = AIEngine(repo)
        response = engine.answer_question("What is PCI DSS?", ["doc1"])
    """

    def __init__(self, repository: DocumentRepository):
        """
        Initializes the AI engine with a document repository.
        Repository is injected to allow testing with mock stores.
        """
        self._repo = repository
        self._client = None  # lazy initialization

    def _get_client(self):
        """
        Returns Anthropic client — created on first use.
        Lazy init means tests can create AIEngine without
        needing real secrets.
        """
        if self._client is None:
            self._client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        return self._client

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """Build context string from retrieval results."""
        if not results:
            return "No relevant context found."
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result.doc_name} "
                f"| Confidence: {result.confidence_pct}%]\n"
                f"{result.chunk_text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def answer_question(self, question: str, doc_ids: list[str]) -> AIResponse:
        """Answer a question using RAG."""
        with Timer() as t:
            try:
                results = self._repo.search(question, doc_ids)
                context = self._build_context(results)

                response = self._get_client().messages.create(
                    model=AI_MODEL,
                    max_tokens=MAX_TOKENS,
                    system="""You are a senior compliance analyst \
for financial services with expertise in PCI DSS, SOX, HIPAA, \
and ADA/WCAG. Answer questions accurately based ONLY on the \
provided context. Always cite which source document your answer \
comes from. If the answer is not in the context, say exactly: \
'I don't have that information in the uploaded documents.' \
Never guess or make up information.""",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Answer using ONLY the context below.

Context:
{context}

Question: {question}""",
                        }
                    ],
                )

                answer = response.content[0].text
                token_count = response.usage.input_tokens + response.usage.output_tokens
                top_confidence = results[0].confidence_pct if results else 0.0

                logger.log_query(
                    question_length=len(question),
                    num_docs=len(doc_ids),
                    confidence=top_confidence,
                    duration_ms=t.duration_ms,
                    token_count=token_count,
                )

                return AIResponse(
                    answer=answer,
                    token_count=token_count,
                    top_confidence=top_confidence,
                    sources=results,
                )

            except Exception as e:
                logger.log_error("answer_question_failed", e)
                raise ValueError(f"Failed to get answer: {str(e)}")

    def analyze_gaps(
        self, doc_id_1: str, doc_name_1: str, doc_id_2: str, doc_name_2: str
    ) -> AIResponse:
        """Compare two documents and identify compliance gaps."""
        with Timer() as t:
            try:
                results1 = self._repo.search(
                    "requirements obligations shall must", [doc_id_1], n_results=5
                )
                results2 = self._repo.search(
                    "requirements obligations shall must", [doc_id_2], n_results=5
                )

                context1 = self._build_context(results1)
                context2 = self._build_context(results2)

                response = self._get_client().messages.create(
                    model=AI_MODEL,
                    max_tokens=2048,
                    system="""You are a senior compliance analyst. \
Analyze documents and identify gaps with precision. \
Structure your response clearly with sections and bullet points.""",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Compare these two compliance \
documents and identify gaps.

DOCUMENT 1 — {doc_name_1}:
{context1}

DOCUMENT 2 — {doc_name_2}:
{context2}

Provide a structured analysis with:
1. **Requirements in {doc_name_2} NOT addressed in {doc_name_1}**
2. **Requirements in {doc_name_1} NOT in {doc_name_2}**
3. **Overall compliance risk** (High / Medium / Low with reason)
4. **Top 3 recommended actions**""",
                        }
                    ],
                )

                answer = response.content[0].text
                token_count = response.usage.input_tokens + response.usage.output_tokens

                logger.info(
                    "gap_analysis_complete",
                    doc1=doc_name_1,
                    doc2=doc_name_2,
                    token_count=token_count,
                    duration_ms=t.duration_ms,
                )

                return AIResponse(
                    answer=answer,
                    token_count=token_count,
                    top_confidence=0.0,
                    sources=results1 + results2,
                )

            except Exception as e:
                logger.log_error("gap_analysis_failed", e)
                raise ValueError(f"Gap analysis failed: {str(e)}")

    def generate_checklist(self, doc_id: str, doc_name: str) -> AIResponse:
        """Generate checklist using top-k retrieval."""
        with Timer() as t:
            try:
                results = self._repo.search(
                    "shall must required mandatory obligated", [doc_id], n_results=5
                )
                context = self._build_context(results)

                response = self._get_client().messages.create(
                    model=AI_MODEL,
                    max_tokens=2048,
                    system="""You are a compliance officer. \
Extract requirements and format them as actionable \
checklist items.""",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Extract ALL compliance \
requirements from this document and format as a checklist.

Document: {doc_name}

Context:
{context}

Format each item as:
- [ ] Requirement description""",
                        }
                    ],
                )

                answer = response.content[0].text
                token_count = response.usage.input_tokens + response.usage.output_tokens

                logger.info(
                    "checklist_generated",
                    doc_name=doc_name,
                    token_count=token_count,
                    duration_ms=t.duration_ms,
                )

                return AIResponse(
                    answer=answer,
                    token_count=token_count,
                    top_confidence=0.0,
                    sources=results,
                )

            except Exception as e:
                logger.log_error("checklist_generation_failed", e)
                raise ValueError(f"Checklist generation failed: {str(e)}")

    def generate_checklist_mapreduce(self, doc_id: str, doc_name: str) -> AIResponse:
        """
        Map-reduce checklist generation.
        Sees ALL chunks — not just top-k.

        MAP:    extract requirements from each batch of chunks
        REDUCE: merge all partial checklists into one final list
        """
        with Timer() as t:
            try:
                collection = self._repo._collections.get(doc_id)
                if not collection:
                    raise ValueError(f"Document {doc_id} not found in repository")

                all_data = collection.get()
                all_chunks = all_data["documents"]
                total_chunks = len(all_chunks)

                logger.info(
                    "mapreduce_started", doc_name=doc_name, total_chunks=total_chunks
                )

                if total_chunks == 0:
                    raise ValueError("No chunks found for document")

                batch_size = 10
                partial_checklists = []
                total_tokens = 0

                for i in range(0, total_chunks, batch_size):
                    batch = all_chunks[i : i + batch_size]
                    batch_text = "\n\n---\n\n".join(batch)
                    batch_num = (i // batch_size) + 1
                    total_batches = (total_chunks + batch_size - 1) // batch_size

                    logger.info(
                        "mapreduce_batch",
                        batch=batch_num,
                        total_batches=total_batches,
                        chunks_in_batch=len(batch),
                    )

                    map_response = self._get_client().messages.create(
                        model=AI_MODEL,
                        max_tokens=1024,
                        system="""You are a compliance analyst.
Extract ONLY explicit requirements from the text.
Look for: shall, must, required, mandatory, obligated.
Format each as: - [ ] Requirement description
If no requirements found, respond with: NO_REQUIREMENTS""",
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Extract all compliance \
requirements from this text.

Document: {doc_name} (batch {batch_num}/{total_batches})

Text:
{batch_text}

List each requirement as:
- [ ] Requirement description""",
                            }
                        ],
                    )

                    total_tokens += (
                        map_response.usage.input_tokens
                        + map_response.usage.output_tokens
                    )
                    result = map_response.content[0].text.strip()
                    if result != "NO_REQUIREMENTS" and result:
                        partial_checklists.append(result)

                if not partial_checklists:
                    return AIResponse(
                        answer="No compliance requirements found " "in this document.",
                        token_count=0,
                        top_confidence=0.0,
                        sources=[],
                    )

                combined = "\n\n".join(partial_checklists)

                reduce_response = self._get_client().messages.create(
                    model=AI_MODEL,
                    max_tokens=2048,
                    system="""You are a senior compliance analyst.
Merge requirement checklists into one clean final list.
Remove duplicates. Group by category if possible.
Keep the - [ ] format for each item.""",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Merge these partial checklists \
into one complete deduplicated compliance checklist.

Document: {doc_name}

Partial checklists:
{combined}

Produce one final organized checklist.
Remove duplicates. Group related requirements.
Keep - [ ] format.""",
                        }
                    ],
                )

                final_checklist = reduce_response.content[0].text
                total_tokens += (
                    reduce_response.usage.input_tokens
                    + reduce_response.usage.output_tokens
                )

                logger.info(
                    "mapreduce_complete",
                    doc_name=doc_name,
                    batches_processed=len(partial_checklists),
                    total_chunks=total_chunks,
                    duration_ms=t.duration_ms,
                )

                return AIResponse(
                    answer=final_checklist,
                    token_count=total_tokens,
                    top_confidence=0.0,
                    sources=[],
                )

            except Exception as e:
                logger.log_error("mapreduce_checklist_failed", e, doc_name=doc_name)
                raise ValueError(f"Map-reduce checklist failed: {str(e)}")
