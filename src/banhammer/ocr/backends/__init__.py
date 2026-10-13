from .base import OCRBackend, OCRResult


def get_backend(name: str, **kw) -> OCRBackend:
    if name == "tesseract":
        from .tesseract import TesseractBackend
        return TesseractBackend(**kw)
    raise ValueError(f"unknown OCR backend: {name}")


__all__ = ["OCRBackend", "OCRResult", "get_backend"]
