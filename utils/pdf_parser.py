"""
utils/pdf_parser.py
-------------------
Extracts text and images from research paper PDFs.
"""

import os
import fitz  # PyMuPDF
from typing import Dict, List

class PDFParser:
    def __init__(self):
        pass

    def extract_text(self, pdf_path: str) -> str:
        """
        Extracts raw text from a PDF file.
        """
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text.strip()

    def extract_images(self, pdf_path: str, output_dir: str = "data/temp/images") -> Dict[str, str]:
        """
        Extracts all images from a PDF and saves them in an output folder.
        Returns a dictionary mapping placeholder names to image paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        image_map = {}
        with fitz.open(pdf_path) as doc:
            img_count = 0
            for i, page in enumerate(doc):
                for img in page.get_images(full=True):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    img_name = f"Figure_{img_count + 1}.png"
                    img_path = os.path.join(output_dir, img_name)
                    if pix.n < 5:  # RGB or grayscale
                        pix.save(img_path)
                    else:  # CMYK
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        pix1.save(img_path)
                        pix1 = None
                    pix = None
                    image_map[f"<<{img_name}>>"] = img_path
                    img_count += 1
        return image_map

    def parse_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        Parses both text and image data from the PDF.
        """
        text = self.extract_text(pdf_path)
        image_map = self.extract_images(pdf_path)
        return {"text": text, "images": image_map}


# ✅ Wrapper function for backward compatibility (must be outside the class)
def extract_text_from_pdf(pdf_path: str) -> str:
    parser = PDFParser()
    return parser.extract_text(pdf_path)
