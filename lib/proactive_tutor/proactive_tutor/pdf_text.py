"""Extract readable text from PDF bytes for the execution agent's read tool.

The desktop runtime reads the file through its permission-checked file tools
and sends the bytes here, so this module never opens filesystem paths.
"""

import io

from pypdf import PdfReader
from pypdf.errors import PdfReadError

# Large enough for book-length PDFs; bounds memory for a single request.
MAX_PDF_BYTES = 200 * 1024 * 1024

NO_TEXT_NOTE = (
    "[No extractable text on this page. It may be a scanned image; "
    "ask the user for an OCR'd copy or a screenshot if its content matters.]"
)


class PdfExtractionError(ValueError):
    """The bytes are not a PDF this service can extract text from."""


def extract_pdf_text(
    data: bytes, first_page: int = 1, last_page: int | None = None
) -> dict:
    """Return page texts for the inclusive 1-indexed range [first_page, last_page]."""
    if len(data) > MAX_PDF_BYTES:
        raise PdfExtractionError(
            f"PDF is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB."
        )
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise PdfExtractionError("PDF is password-protected.")
        total = len(reader.pages)
    except PdfExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError) as error:
        raise PdfExtractionError(f"Could not parse PDF: {error}") from error

    if total == 0:
        return {"total_pages": 0, "first_page": 0, "last_page": 0, "pages": []}
    if first_page < 1 or first_page > total:
        raise PdfExtractionError(
            f"Page {first_page} is out of range (PDF has {total} pages)."
        )
    last = total if last_page is None else min(max(last_page, first_page), total)

    pages = []
    for number in range(first_page, last + 1):
        try:
            text = (reader.pages[number - 1].extract_text() or "").strip()
        except Exception as error:  # pypdf raises many types on malformed pages
            text = f"[Could not extract text from this page: {error}]"
        pages.append({"number": number, "text": text or NO_TEXT_NOTE})
    return {
        "total_pages": total,
        "first_page": first_page,
        "last_page": last,
        "pages": pages,
    }
