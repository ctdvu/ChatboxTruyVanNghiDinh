"""Tạo chỉ mục RAG nhẹ từ các PDF trong thư mục data/."""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pdfplumber
import fitz
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "data"
DEFAULT_INDEX = ROOT / "storage" / "rag_index.joblib"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_text(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    """Cắt theo độ dài ký tự, ưu tiên dừng ở cuối câu/khoản."""
    text = clean_text(text)
    parts, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            pivot = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind("\n", start, end))
            if pivot > start + size // 2:
                end = pivot + 1
        piece = text[start:end].strip()
        if len(piece) >= 80:
            parts.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return parts


def extract_page_text(pdf_path: Path, page_no: int, fallback: str) -> str:
    """Dùng Tesseract cho PDF scan không có lớp văn bản."""
    if fallback.strip():
        return fallback
    document = fitz.open(pdf_path)
    try:
        page = document[page_no - 1]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        try:
            return pytesseract.image_to_string(Image.open(BytesIO(pixmap.tobytes("png"))), lang="vie+eng")
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Chưa cài Tesseract OCR. Cài từ https://github.com/UB-Mannheim/tesseract/wiki, "
                "thêm C:\\Program Files\\Tesseract-OCR vào PATH rồi khởi động lại Jupyter."
            ) from exc
    finally:
        document.close()


def build_index(pdf_dir: Path = PDF_DIR, output: Path = DEFAULT_INDEX) -> dict:
    records = []
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Không có PDF trong {pdf_dir}")
    for pdf_path in pdfs:
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                text = extract_page_text(pdf_path, page_no, page.extract_text() or "")
                for chunk_no, chunk in enumerate(split_text(text), start=1):
                    records.append({"text": chunk, "source": pdf_path.name, "page": page_no, "chunk": chunk_no})
    if not records:
        raise ValueError("Không đọc được nội dung PDF; hãy kiểm tra PDF có lớp văn bản.")
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, max_df=0.98)
    matrix = vectorizer.fit_transform([r["text"] for r in records])
    payload = {"vectorizer": vectorizer, "matrix": matrix, "records": records,
               "created_at": datetime.now(timezone.utc).isoformat(), "pdf_count": len(pdfs)}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output)
    return payload


if __name__ == "__main__":
    result = build_index()
    print(f"Đã tạo {len(result['records'])} đoạn từ {result['pdf_count']} PDF: {DEFAULT_INDEX}")
