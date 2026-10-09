"""Utilities for converting raw document formats (PDF) to images."""

from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from system.logging import get_logger

logger = get_logger(__name__)


def enhance_receipt_image(img: Image.Image, aggressive: bool = False) -> Image.Image:
    """Apply OCR-optimized preprocessing to receipt images.

    Args:
        img: Input PIL Image
        aggressive: If True, applies stronger enhancement and optional binarization
                    for extremely faded receipts

    Returns:
        Enhanced PIL Image optimized for OCR
    """
    # Ensure RGB input
    if img.mode != "RGB":
        img = img.convert("RGB")

    if CV2_AVAILABLE:
        # PIL → OpenCV
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 1. Grayscale (OCR expects luminance, not color)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 2. Light denoising (preserve thin character strokes)
        denoised = cv2.fastNlMeansDenoising(gray, h=12 if aggressive else 8)

        # 3. CLAHE — core receipt enhancer
        clahe = cv2.createCLAHE(clipLimit=3.0 if aggressive else 2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # 4. OPTIONAL binarization (only for extremely faded receipts)
        # Modern OCR prefers grayscale — thresholding is a fallback, not default
        if aggressive:
            enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Return grayscale image (preferred by OCR)
        result = Image.fromarray(enhanced)

    else:
        # PIL-only fallback (best-effort)
        logger.warning(
            "OpenCV not available, using basic PIL preprocessing. "
            "Install opencv-python for better OCR results."
        )

        gray = img.convert("L")

        # Mild contrast boost (avoid hard clipping)
        enhancer = ImageEnhance.Contrast(gray)
        contrasted = enhancer.enhance(1.8 if aggressive else 1.4)

        # Optional binarization only in aggressive mode
        if aggressive:
            threshold = 140
            contrasted = contrasted.point(lambda x: 255 if x > threshold else 0, mode="L")

        result = contrasted

    return result


def convert_pdf_to_images(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 300,
    enhance_for_ocr: bool = True,
    aggressive_enhancement: bool = False,
) -> List[Path]:
    """Convert multi-page PDF into individual images.

    Args:
        pdf_path: Path to the source PDF.
        output_dir: Directory where processed images will be saved.
        dpi: Resolution for the output images (300+ recommended for OCR).
        enhance_for_ocr: Apply receipt-optimized preprocessing.
        aggressive_enhancement: Use stronger enhancement for very faded receipts.

    Returns:
        List of paths to the generated images.
    """
    if fitz is None:
        logger.error("PyMuPDF (fitz) not installed. Run 'pip install pymupdf'")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
            output_path = output_dir / f"{pdf_path.stem}_p{page_num:03d}.jpg"
            
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if enhance_for_ocr:
                img = enhance_receipt_image(img, aggressive=aggressive_enhancement)

            img.save(output_path, "JPEG", quality=95)
            generated_paths.append(output_path)

        doc.close()
        logger.info(f"Converted {pdf_path.name} to {len(generated_paths)} images")

    except Exception as e:
        logger.error(f"Failed to convert PDF {pdf_path}: {e}")

    return generated_paths


def preprocess_raw_image_data(
    source_path: Path,
    batch_id: str,
    artifacts_root: Path,
    enhance_for_ocr: bool = True,
    aggressive_enhancement: bool = False,
) -> List[Path]:
    """Handle raw document ingestion (Image or PDF).

    Creates a structured directory: /data/processed/{batch_id}/

    Args:
        source_path: Path to source document
        batch_id: Unique identifier for this batch
        artifacts_root: Root directory for processed files
        enhance_for_ocr: Apply receipt-optimized preprocessing
        aggressive_enhancement: Use stronger enhancement for very faded receipts
    """
    processed_dir = artifacts_root / "processed" / batch_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    if source_path.suffix.lower() == ".pdf":
        return convert_pdf_to_images(
            source_path,
            processed_dir,
            enhance_for_ocr=enhance_for_ocr,
            aggressive_enhancement=aggressive_enhancement,
        )

    elif source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        # For single images, process and save
        dest_path = processed_dir / f"{source_path.stem}.jpg"
        img = Image.open(source_path).convert("RGB")

        # Apply OCR preprocessing if requested
        if enhance_for_ocr:
            img = enhance_receipt_image(img, aggressive=aggressive_enhancement)

        img.save(dest_path, "JPEG", quality=95)
        return [dest_path]

    else:
        logger.warning(f"Unsupported file format: {source_path.suffix}")
        return []
