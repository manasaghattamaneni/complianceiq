# core/ai_engine.py
# AI Engine — all Claude API interactions in one place
# Single Responsibility: ONLY handles AI calls
# Uses retrieval results to build grounded prompts

import anthropic
import streamlit as st
from dataclasses import dataclass
from config import AI_MODEL, MAX_TOKENS
from core.retrieval import DocumentRepository, RetrievalResult
from utils.logger import logger
from utils.metrics import Timer, QueryMetric


@dataclass
class AIResponse:
    """
    Structured response from Claude API.
    Clean DTO — callers get typed object not raw API response.
    """

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
        self._client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    def _build_context(self, results: list[RetrievalResult]) -> str:
        """
        Build context string from retrieval results.
        Each chunk labeled with source and confidence.
        """
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
        """
        Answer a question using RAG.
        1. Retrieve relevant chunks
        2. Build context
        3. Call Claude
        4. Return structured response
        """
        with Timer() as t:
            try:
                # Step 1 — Retrieve relevant chunks
                results = self._repo.search(question, doc_ids)

                # Step 2 — Build context from results
                context = self._build_context(results)

                # Step 3 — Call Claude with context
                response = self._client.messages.create(
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
                            "content": f"""Answer using ONLY the \
context below.

Context:
{context}

Question: {question}""",
                        }
                    ],
                )

                answer = response.content[0].text
                token_count = response.usage.input_tokens + response.usage.output_tokens
                top_confidence = results[0].confidence_pct if results else 0.0

                # Log query metrics
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
        """
        Compare two documents and identify compliance gaps.
        Retrieves broad context from both docs then asks
        Claude to identify missing requirements.
        """
        with Timer() as t:
            try:
                # Get broad context from both documents
                results1 = self._repo.search(
                    "requirements obligations shall must", [doc_id_1], n_results=5
                )
                results2 = self._repo.search(
                    "requirements obligations shall must", [doc_id_2], n_results=5
                )

                context1 = self._build_context(results1)
                context2 = self._build_context(results2)

                response = self._client.messages.create(
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
1. **Requirements in {doc_name_2} NOT addressed in {doc_name_1}** \
(critical gaps)
2. **Requirements in {doc_name_1} NOT in {doc_name_2}** \
(additional controls)
3. **Overall compliance risk** (High / Medium / Low with reason)
4. **Top 3 recommended actions** (specific and actionable)

Be precise. Reference specific requirements where possible.""",
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
        """
        Auto-generate compliance checklist from a document.
        Extracts all 'shall', 'must', 'required' statements
        and formats them as actionable checklist items.
        """
        with Timer() as t:
            try:
                # Search specifically for requirement language
                results = self._repo.search(
                    "shall must required mandatory obligated", [doc_id], n_results=5
                )

                context = self._build_context(results)

                response = self._client.messages.create(
                    model=AI_MODEL,
                    max_tokens=2048,
                    system="""You are a compliance officer. \
Extract requirements and format them as actionable \
checklist items. Be thorough and specific.""",
                    messages=[
                        {
                            "role": "user",
                            "content": f"""Extract ALL compliance \
requirements from this document and format as a checklist.

Document: {doc_name}

Context:
{context}

Format each item as:
- [ ] Requirement description (Section reference if available)

Group by category if possible. Include every 'shall', \
'must', 'required', 'mandatory' statement.""",
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
