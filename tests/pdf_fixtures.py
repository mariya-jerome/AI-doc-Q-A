"""Minimal PDF byte builders for unit tests (no reportlab dependency)."""

from __future__ import annotations


def _pdf_with_page_stream(content_stream: bytes) -> bytes:
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        b"4 0 obj\n<< /Length %d >>\nstream\n" % len(content_stream) + content_stream + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    body = b""
    offsets = [0]
    position = len(header)
    for obj in objects:
        offsets.append(position)
        body += obj
        position += len(obj)

    xref_start = position
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets[1:]:
        xref += f"{offset:010d} 00000 n \n".encode("ascii")
    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
        + str(xref_start).encode("ascii")
        + b"\n%%EOF\n"
    )
    return header + body + xref + trailer


def pdf_with_text(text: str) -> bytes:
    """Build a one-page PDF whose extractable text includes `text`."""

    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", errors="replace")
    return _pdf_with_page_stream(stream)


def pdf_without_text() -> bytes:
    """Build a one-page PDF with no text operators."""

    return _pdf_with_page_stream(b"")
