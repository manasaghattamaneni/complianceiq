import fitz
import pandas as pd
from docx import Document
from utils.logger import logger
from utils.metrics import Timer


def _read_pdf(file) -> tuple[str, int]:
    pdf_bytes = file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    num_pages = len(doc)

    for page in doc:
        text = page.get_text()
        if len(text.strip()) < 50:
            blocks = page.get_text("blocks")
            text = "\n".join([b[4] for b in blocks if b[4].strip()])
        full_text += text + "\n"

    doc.close()
    return full_text, num_pages


def _read_docx(file) -> tuple[str, int]:
    """
    Extract text from Word document using python-docx.
    Returns: (full_text, page_count)
    Note: DOCX has no real page count — we estimate from word count.
    """
    doc = Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n\n".join(paragraphs)

    # Estimate pages: average 300 words per page
    word_count = len(full_text.split())
    estimated_pages = max(1, word_count // 300)

    return full_text, estimated_pages


def _read_txt(file) -> tuple[str, int]:
    """
    Extract text from plain text file.
    Returns: (full_text, estimated_pages)
    """
    content = file.read()

    # Handle different encodings gracefully
    if isinstance(content, bytes):
        try:
            full_text = content.decode("utf-8")
        except UnicodeDecodeError:
            full_text = content.decode("latin-1")
    else:
        full_text = content

    word_count = len(full_text.split())
    estimated_pages = max(1, word_count // 300)

    return full_text, estimated_pages


def _read_csv(file) -> tuple[str, int]:
    """
    Extract text from CSV using pandas.
    Converts table to readable text format.
    Returns: (full_text, row_count_as_pages)
    """
    df = pd.read_csv(file)

    lines = []
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append(f"Total rows: {len(df)}")
    lines.append("")

    for _, row in df.iterrows():
        row_text = " | ".join(
            [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        )
        lines.append(row_text)

    full_text = "\n".join(lines)
    estimated_pages = max(1, len(df) // 50)

    return full_text, estimated_pages


def extract_text(file) -> tuple[str, int, str]:
    """
    Factory function — decides which reader to use.
    Returns: (full_text, page_count, file_type)
    """
    with Timer() as t:
        try:
            extension = file.name.lower().split(".")[-1]

            readers = {
                "pdf": _read_pdf,
                "docx": _read_docx,
                "txt": _read_txt,
                "csv": _read_csv,
            }

            if extension not in readers:
                raise ValueError(f"Unsupported file type: .{extension}")

            reader_function = readers[extension]
            full_text, pages = reader_function(file)
            file_type = extension

            if len(full_text.strip()) < 100:
                raise ValueError(
                    "Could not extract enough readable text. "
                    "The document may be scanned, "
                    "image-based, or empty."
                )

            doc_name = file.name
            logger.log_upload(
                doc_name=doc_name,
                file_type=file_type,
                pages=pages,
                chunks=0,
                duration_ms=t.duration_ms,
            )

            return full_text, pages, file_type

        except ValueError:
            raise

        except Exception as e:
            doc_name = file.name if file else "unknown"
            logger.log_error("ingestion_failed", e, doc_name=doc_name)
            raise ValueError(f"Failed to read document: {str(e)}")
