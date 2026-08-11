"""Common interface over OCR engines used by the text_ocr extractor family.

Each function takes an image path and returns plain extracted text. We
deliberately return plain text (not bounding boxes) here — the point of
this benchmark's text_ocr family is "how well does OCR-text + SLM compare
to a VLM reading the pixels directly", not full layout-aware IE. If you
want layout-aware text_ocr, see docs/ARCHITECTURE.md for how to extend
`ocr_text` on DocumentSample with bounding-box-annotated text instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def run_ocr(engine: str, image_path: Path) -> str:
    if engine == "tesseract":
        return _tesseract(image_path)
    if engine == "paddleocr":
        return _paddleocr(image_path)
    if engine == "easyocr":
        return _easyocr(image_path)
    raise ValueError(f"Unknown OCR engine '{engine}'. See configs/models.yaml.")


def _tesseract(image_path: Path) -> str:
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(Image.open(image_path))


def _paddleocr(image_path: Path) -> str:
    reader = _paddle_reader()
    result = reader.ocr(str(image_path), cls=True)
    lines = [line[1][0] for page in result for line in page]
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _paddle_reader():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def _easyocr(image_path: Path) -> str:
    reader = _easyocr_reader()
    result = reader.readtext(str(image_path), detail=0)
    return "\n".join(result)


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False)
