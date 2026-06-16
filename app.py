import streamlit as st
import time
import hashlib
import uuid
from config import APP_NAME, APP_VERSION, RATE_LIMIT_SECONDS
from core.security import SecurityValidator, SecurityError
from core.ingestion import extract_text
from core.chunking import split_into_chunks
from core.retrieval import DocumentRepository
from core.ai_engine import AIEngine
from utils.logger import logger
from utils.metrics import SessionMetrics, QueryMetric

st.set_page_config(page_title=APP_NAME, page_icon="🔍", layout="centered")


def init_session():
    try:
        _ = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        st.error(
            "🔒 Missing API key. Add ANTHROPIC_API_KEY " "to .streamlit/secrets.toml"
        )
        st.stop()

    # Guarded init — only runs once per session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "repo" not in st.session_state:
        st.session_state.repo = DocumentRepository()
        if "documents" not in st.session_state:
            st.session_state.documents = {}
        restored = st.session_state.repo.restore_collections()
        for i, doc in enumerate(restored):
            slot = f"doc{i + 1}"
            display_name = doc["doc_name"]
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."
            st.session_state.documents[slot] = {
                "id": doc["doc_id"],
                "name": doc["doc_name"],
                "display_name": display_name,
                "pages": doc["pages"],
                "chunks": doc["chunks"],
                "type": doc["file_type"].upper(),
            }
        if restored:
            logger.info("ui_documents_restored", count=len(restored))

    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = SessionMetrics()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}
    if "last_request" not in st.session_state:
        st.session_state.last_request = 0.0
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    if st.session_state.engine is None:
        st.session_state.engine = AIEngine(st.session_state.repo)


init_session()


def reset_session_for_new_document():
    """Full reset when new document uploaded."""
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.feedback = {}
    st.session_state.metrics.reset()
    logger.info("session_reset", session_id=st.session_state.session_id)


def clear_all_documents():
    st.session_state.repo.clear_all()
    st.session_state.documents = {}
    st.session_state.uploader_key += 1  # forces uploader reset
    reset_session_for_new_document()
    logger.info("all_documents_cleared")


def process_upload(uploaded_file, doc_slot: str):
    """Handle document upload end to end."""
    try:
        SecurityValidator.validate_file(uploaded_file)

        with st.spinner(f"Reading {uploaded_file.name}..."):
            text, pages, file_type = extract_text(file=uploaded_file)

        with st.spinner("Indexing document..."):
            chunks = split_into_chunks(text, file_type=file_type)
            # Stable doc_id based on content hash — not session_id
            # Same document content always gets the same doc_id
            # Re-uploading replaces the old collection cleanly
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            doc_id = f"{doc_slot}_{content_hash}"
            st.session_state.repo.store_document(
                doc_id, chunks, uploaded_file.name, pages=pages, file_type=file_type
            )

        display_name = uploaded_file.name
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        st.session_state.documents[doc_slot] = {
            "id": doc_id,
            "name": uploaded_file.name,
            "display_name": display_name,
            "pages": pages,
            "chunks": len(chunks),
            "type": file_type.upper(),
        }

        reset_session_for_new_document()
        st.success(f"✅ Indexed {len(chunks)} chunks " f"across {pages} pages")
        st.rerun()

    except SecurityError as e:
        st.error(f"🔒 {str(e)}")
    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error("❌ Unexpected error. Please try again.")
        logger.log_error("upload_failed", e, doc_name=uploaded_file.name)


def record_task_metric(response, duration_ms: float):
    """Record a one-shot engine task (gap analysis / checklist) in metrics."""
    st.session_state.metrics.add_query(
        QueryMetric(
            question_length=0,
            duration_ms=duration_ms,
            token_count=response.token_count,
            confidence_pct=response.top_confidence,
        )
    )


with st.sidebar:
    st.markdown(f"### 📊 Session Metrics")

    m = st.session_state.metrics
    docs = st.session_state.documents

    if docs:
        st.markdown("**Documents loaded**")
        for slot, doc in docs.items():
            st.markdown(
                f"- {doc['display_name']} "
                f"({doc['type']} · {doc['pages']}p · "
                f"{doc['chunks']} chunks)"
            )
        st.divider()

    # Tokens tracked internally in logs only
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions asked", m.total_queries)
    with col2:
        st.metric("Helpful answers", m.feedback_ratio)

    logger.info(
        "session_metrics_snapshot",
        avg_latency_ms=m.avg_latency_ms,
        avg_confidence=m.avg_confidence,
        total_queries=m.total_queries,
    )

    if st.session_state.messages:
        st.divider()
        log_lines = []
        for msg in st.session_state.messages:
            role = "Q" if msg["role"] == "user" else "A"
            log_lines.append(f"{role}: {msg['content']}\n")
        log_text = "\n".join(log_lines)
        st.download_button(
            label="📥 Export session",
            data=log_text,
            file_name="complianceiq_session.txt",
            mime="text/plain",
        )

    st.divider()
    st.caption(
        f"{APP_NAME} v{APP_VERSION}\n" "Python · ChromaDB · Claude API · Streamlit"
    )

st.title("🔍 ComplianceIQ")
st.caption(
    "AI-powered compliance document analyzer · "
    "Powered by Claude API · RAG + ChromaDB"
)

st.markdown("### 📄 Documents")
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.markdown("**Document 1**")
    file1 = st.file_uploader(
        "Primary document",
        type=["pdf", "docx", "txt", "csv"],
        key=f"uploader_1_{st.session_state.uploader_key}",
    )
    if file1:
        current = st.session_state.documents.get("doc1", {})
        if file1.name != current.get("name"):
            process_upload(file1, "doc1")

with upload_col2:
    st.markdown("**Document 2** *(optional)*")
    file2 = st.file_uploader(
        "Compare against",
        type=["pdf", "docx", "txt", "csv"],
        key=f"uploader_2_{st.session_state.uploader_key}",
    )
    if file2:
        current = st.session_state.documents.get("doc2", {})
        if file2.name != current.get("name"):
            process_upload(file2, "doc2")

if len(st.session_state.documents) >= 2:
    st.divider()
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("🔍 Analyze Gaps", type="primary", use_container_width=True):
            docs_list = list(st.session_state.documents.values())
            doc1 = docs_list[0]
            doc2 = docs_list[1]
            with st.spinner("Analyzing compliance gaps..."):
                try:
                    start = time.perf_counter()
                    response = st.session_state.engine.analyze_gaps(
                        doc1["id"], doc1["name"], doc2["id"], doc2["name"]
                    )
                    duration_ms = (time.perf_counter() - start) * 1000
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"🔍 Gap Analysis: "
                            f"{doc1['display_name']} vs "
                            f"{doc2['display_name']}",
                        }
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.answer}
                    )
                    record_task_metric(response, duration_ms)
                    logger.info("gap_analysis_tokens", tokens=response.token_count)
                    st.rerun()
                except Exception as e:
                    st.session_state.metrics.add_error()
                    st.error("❌ Gap analysis failed. Please try again.")
                    logger.log_error("gap_analysis_ui_failed", e)

    with btn_col2:
        if st.button("📋 Generate Checklist", use_container_width=True):
            doc1 = st.session_state.documents["doc1"]
            with st.spinner("Generating compliance checklist..."):
                try:
                    start = time.perf_counter()
                    response = st.session_state.engine.generate_checklist_mapreduce(
                        doc1["id"], doc1["name"]
                    )
                    duration_ms = (time.perf_counter() - start) * 1000
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"📋 Checklist: {doc1['display_name']}",
                        }
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.answer}
                    )
                    record_task_metric(response, duration_ms)
                    st.rerun()
                except Exception as e:
                    st.session_state.metrics.add_error()
                    st.error("❌ Checklist generation failed. Please try again.")
                    logger.log_error("checklist_ui_failed", e)

    with btn_col3:
        if st.button("🗑️ Clear Documents", use_container_width=True):
            clear_all_documents()
            st.rerun()

elif len(st.session_state.documents) == 1:
    st.divider()
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("📋 Generate Checklist", type="primary", use_container_width=True):
            doc1 = list(st.session_state.documents.values())[0]
            with st.spinner("Generating compliance checklist..."):
                try:
                    start = time.perf_counter()
                    response = st.session_state.engine.generate_checklist(
                        doc1["id"], doc1["name"]
                    )
                    duration_ms = (time.perf_counter() - start) * 1000
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"📋 Checklist: {doc1['display_name']}",
                        }
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.answer}
                    )
                    record_task_metric(response, duration_ms)
                    st.rerun()
                except Exception as e:
                    st.session_state.metrics.add_error()
                    st.error("❌ Checklist generation failed. Please try again.")
                    logger.log_error("checklist_ui_failed", e)

    with btn_col2:
        if st.button("🗑️ Clear Documents", use_container_width=True):
            clear_all_documents()
            st.rerun()

if st.session_state.documents:
    st.divider()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant":
                fb_key = f"fb_{i}"
                current_fb = st.session_state.feedback.get(fb_key)

                fc1, fc2, _ = st.columns([1, 1, 8])
                with fc1:
                    if st.button(
                        "👍",
                        key=f"up_{i}",
                        type="primary" if current_fb == "up" else "secondary",
                    ):
                        st.session_state.feedback[fb_key] = "up"
                        st.session_state.metrics.sync_feedback(
                            st.session_state.feedback.values()
                        )
                        logger.log_feedback("up", 0)
                        st.rerun()
                with fc2:
                    if st.button(
                        "👎",
                        key=f"down_{i}",
                        type="primary" if current_fb == "down" else "secondary",
                    ):
                        st.session_state.feedback[fb_key] = "down"
                        st.session_state.metrics.sync_feedback(
                            st.session_state.feedback.values()
                        )
                        logger.log_feedback("down", 0)
                        st.rerun()

    question = st.chat_input("Ask a question about your document(s)...")

    if question:
        now = time.time()
        if now - st.session_state.last_request < RATE_LIMIT_SECONDS:
            st.warning("Please wait a moment before asking again.")
            st.stop()
        st.session_state.last_request = now

        try:
            clean_question = SecurityValidator.validate_question(question)
        except SecurityError as e:
            st.error(f"🔒 {str(e)}")
            st.stop()

        doc_ids = [doc["id"] for doc in st.session_state.documents.values()]

        st.session_state.messages.append({"role": "user", "content": clean_question})

        try:
            start = time.perf_counter()

            response = st.session_state.engine.answer_question(clean_question, doc_ids)
            duration_ms = (time.perf_counter() - start) * 1000

            st.session_state.messages.append(
                {"role": "assistant", "content": response.answer}
            )

            st.session_state.metrics.add_query(
                QueryMetric(
                    question_length=len(clean_question),
                    duration_ms=duration_ms,
                    token_count=response.token_count,
                    confidence_pct=response.top_confidence,
                )
            )

            logger.info("query_tokens", tokens=response.token_count)

        except Exception as e:
            st.session_state.metrics.add_error()
            logger.log_error("answer_question_ui_failed", e)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "❌ Something went wrong answering that. "
                    "Please try again.",
                }
            )

        # Single rerun — everything saved before this fires
        st.rerun()

else:
    st.info("👆 Upload a document above to get started")
    st.markdown("""
    **What you can do:**
    - 📄 Upload PDF, Word, CSV, or TXT documents
    - 💬 Ask questions in plain English
    - 📄📄 Upload two documents and compare them
    - 🔍 Click **Analyze Gaps** to find compliance gaps
    - 📋 Click **Generate Checklist** to extract requirements
    - 👍👎 Rate answers to track quality
    - 📥 Export your session as a text file
    """)
