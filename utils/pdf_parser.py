"""
Parse PDF into page-wise text and extract images (with bbox + page info).
Primary backend: PyMuPDF (fitz).
Optional: pytesseract OCR fallback for scanned PDFs (if installed).

Returns:
- pages: List[(page_number:int, text:str)]
- images: List[agents.ImageRef]
"""
from typing import List, Tuple, Optional
import os
import io
import hashlib
import logging

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise ImportError("PyMuPDF (fitz) is required. Install with: pip install pymupdf") from e

try:
    from PIL import Image
except Exception:
    raise ImportError("Pillow is required. Install with: pip install pillow")

# Optional OCR fallback
try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Import ImageRef dataclass from agents package (canonical type)
try:
    from agents import ImageRef
except Exception:
    from dataclasses import dataclass, field
    from typing import Dict, Any, Optional, List

    @dataclass
    class ImageRef:
        id: str
        page: int
        bbox: Optional[List[float]] = None
        caption: Optional[str] = None
        path: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------- Utility helpers ----------------------------
def _make_hash(data: bytes) -> str:
    """Return a short SHA1 hash of given bytes."""
    return hashlib.sha1(data).hexdigest()[:12]


def _ensure_image_output_dir(out_dir: str):
    """Ensure that image output directory exists."""
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    return img_dir


# ---------------------------- Main function ----------------------------
def parse_pdf_to_pages_and_images(
    pdf_path: str,
    output_dir: str = "data/outputs",
    ocr_if_empty: bool = True
) -> Tuple[List[Tuple[int, str]], List[ImageRef]]:
    """
    Parse a PDF and return:
      pages: [(page_no, text), ...]
      images: [ImageRef(...), ...] with .path pointing to saved files.
    """
    logger.info("Parsing PDF: %s", pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    images = []
    img_out_dir = _ensure_image_output_dir(output_dir)

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        page_num = pno + 1

        # Extract structured text
        try:
            text = page.get_text("text")
        except Exception:
            text = page.get_text()

        # If text is too short, use OCR fallback if available
        if (not text or len(text.strip()) < 10) and ocr_if_empty and OCR_AVAILABLE:
            logger.debug("Page %d has little text — attempting OCR", page_num)
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes()))
                ocr_text = pytesseract.image_to_string(img)
                text = ocr_text or text
            except Exception as e:
                logger.debug("OCR failed for page %d: %s", page_num, e)

        pages.append((page_num, text or ""))

        # Extract images from dict representation
        try:
            j = page.get_text("dict")
            blocks = j.get("blocks", [])
        except Exception:
            blocks = []

        img_index = 0
        for b in blocks:
            # block["type"] == 1 → image
            if b.get("type") == 1:
                img_index += 1
                bbox = b.get("bbox", None)
                img_info = b.get("image")
                xref = None
                img_bytes = None
                ext = "png"

                # --- Handle both dict and bytes cases ---
                if isinstance(img_info, dict):
                    xref = img_info.get("xref")
                elif isinstance(img_info, (bytes, bytearray)):
                    img_bytes = img_info
                # ---------------------------------------

                saved_path = None
                try:
                    if xref:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image.get("image")
                        ext = base_image.get("ext", "png")
                    elif img_bytes is None:
                        # fallback render of region
                        mat = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=mat, clip=bbox)
                        img_bytes = pix.tobytes()
                        ext = "png"

                    if img_bytes:
                        hid = _make_hash(img_bytes)
                        filename = f"img_p{page_num}_{img_index}_{hid}.{ext}"
                        saved_path = os.path.join(img_out_dir, filename)
                        with open(saved_path, "wb") as fh:
                            fh.write(img_bytes)

                except Exception as e:
                    logger.debug("Failed extracting image on page %d, block %d: %s", page_num, img_index, e)
                    saved_path = None

                # Find nearby caption
                caption = _guess_caption_for_bbox(blocks, bbox)
                img_id = os.path.basename(saved_path) if saved_path else f"img_p{page_num}_{img_index}"

                images.append(
                    ImageRef(
                        id=img_id,
                        page=page_num,
                        bbox=bbox,
                        caption=caption,
                        path=saved_path,
                        metadata={"xref": xref},
                    )
                )

        # Inline images not captured in blocks
        try:
            page_images = page.get_images(full=True)
            for idx, pimg in enumerate(page_images, start=1):
                xref = pimg[0]
                already = any(img.metadata.get("xref") == xref for img in images)
                if already:
                    continue
                try:
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    ext = base_image.get("ext", "png")
                    hid = _make_hash(img_bytes)
                    filename = f"img_p{page_num}_xref{xref}_{hid}.{ext}"
                    saved_path = os.path.join(img_out_dir, filename)
                    with open(saved_path, "wb") as fh:
                        fh.write(img_bytes)
                    images.append(
                        ImageRef(
                            id=os.path.basename(saved_path),
                            page=page_num,
                            bbox=None,
                            caption=None,
                            path=saved_path,
                            metadata={"xref": xref},
                        )
                    )
                except Exception as e:
                    logger.debug("Failed to write page image xref %s: %s", xref, e)
        except Exception:
            pass

    doc.close()
    logger.info("Parsed %d pages and saved %d images into %s", len(pages), len(images), img_out_dir)
    return pages, images


# ---------------------------- Caption Heuristic ----------------------------
def _guess_caption_for_bbox(blocks: list, bbox: Optional[List[float]]) -> Optional[str]:
    """Find a nearby caption block below an image using simple heuristics."""
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    candidates = []
    for b in blocks:
        if b.get("type") != 0:
            continue
        bb = b.get("bbox", None)
        if not bb:
            continue
        bx0, by0, bx1, by1 = bb
        vertical_gap = by0 - y1
        horiz_overlap = (min(x1, bx1) - max(x0, bx0))
        if vertical_gap >= 0 and vertical_gap < (y1 - y0) * 2 and horiz_overlap > 10:
            txt = b.get("text", "").strip()
            if not txt:
                continue
            if "fig" in txt.lower() or "figure" in txt.lower() or len(txt) < 200:
                candidates.append((vertical_gap, txt))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1].replace("\n", " ").strip()