# pipeline/workflow.py
"""
High-level workflow definitions for the research-paper -> poster pipeline.

Expose a Workflow class with discrete step functions that can be composed, tested or reused.
This file focuses on composition and retry/error policies for each step.
"""

from typing import List, Tuple, Optional
import time
import logging
from functools import wraps

from agents import Section, ImageRef
from utils.pdf_parser import parse_pdf_to_pages_and_images
from utils.text_cleaner import clean_pages_text
from utils.image_handler import associate_images_to_sections

logger = logging.getLogger(__name__)


def retry_on_exception(retries: int = 2, delay: float = 1.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logger.warning("Function %s failed on attempt %d/%d: %s", fn.__name__, attempt + 1, retries + 1, e)
                    time.sleep(delay)
            logger.exception("Function %s failed after %d retries", fn.__name__, retries + 1)
            raise last_exc
        return wrapped
    return deco


class Workflow:
    """
    Encapsulates the high-level steps. Each method returns objects needed for subsequent steps.
    """

    def __init__(self, ocr_if_empty: bool = True):
        self.ocr_if_empty = ocr_if_empty
        self.logger = logging.getLogger("workflow")

    @retry_on_exception(retries=1, delay=1.0)
    def parse_pdf(self, pdf_path: str, output_dir: str = "data/outputs") -> Tuple[List[Tuple[int, str]], List[ImageRef]]:
        return parse_pdf_to_pages_and_images(pdf_path, output_dir=output_dir, ocr_if_empty=self.ocr_if_empty)

    def clean_pages(self, pages: List[Tuple[int, str]], remove_references: bool = True) -> List[Tuple[int, str]]:
        return clean_pages_text(pages, remove_references=remove_references)

    def detect_sections(self, pages: List[Tuple[int, str]], section_identifier) -> List[Section]:
        # section_identifier is expected to be an instance of SectionIdentifier
        return section_identifier.identify_sections_from_pages(pages)

    def map_images(self, sections: List[Section], images: List[ImageRef]) -> List[Section]:
        return associate_images_to_sections(sections, images)

    def summarize_sections(self, sections: List[Section], summarizer) -> List[Section]:
        """
        summarizer: an instance of SummarizerAgent
        Returns sections with summarizer metadata populated (e.g., metadata['bullets']).
        """
        out = []
        for sec in sections:
            try:
                summary = summarizer.summarize_section(sec)
                sec.metadata["bullets"] = summary.get("bullets", [])
                # update image captions if provided
                for img_obj in summary.get("image_refs", []):
                    for real_img in sec.images:
                        if real_img.id == img_obj.get("id"):
                            if img_obj.get("caption"):
                                real_img.caption = img_obj.get("caption")
                out.append(sec)
            except Exception as e:
                logger.warning("Summarization failed for section %s: %s", sec.title, e)
                out.append(sec)
        return out
