# pipeline/pipeline_manager.py
"""
PipelineManager
- Orchestrates the full run: parse PDF -> clean -> identify sections -> map images -> summarize -> format poster
- Writes outputs to configured output_dir (default: data/outputs)
"""

from typing import Optional, List, Tuple
import os
import logging
import yaml
import re

# agent modules
from agents.section_identifier import SectionIdentifier
from agents.summarizer_agent import SummarizerAgent
from agents.poster_formatter import PosterFormatter

# utils
from utils.logger import setup_logging
from utils.pdf_parser import parse_pdf_to_pages_and_images
from utils.text_cleaner import clean_pages_text
from utils.image_handler import associate_images_to_sections
from utils.llm_client import LLMClient  # ✅ using local Qwen client

# dataclasses
from agents import Section, Poster

logger = logging.getLogger(__name__)


class PipelineManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Load configuration and prepare components.
        """
        self.config = self._load_config(config_path)
        self.logger = setup_logging("pipeline_manager")
        self.output_dir = self.config.get("paths", {}).get("outputs_dir", "data/outputs")
        os.makedirs(self.output_dir, exist_ok=True)

                # ✅ Initialize local Qwen LLM client if backend = qwen-local
        backend = self.config.get("llm", {}).get("backend", "none")

        try:
            if backend == "qwen_local" or backend == "qwen-local":
                self.llm_client = LLMClient(model_path=self.config["llm"]["model_path"])
                self.logger.info("✅ LLM client initialized using local Qwen model.")
            else:
                self.llm_client = None
                self.logger.warning(f"No valid LLM backend configured ('{backend}'). Running heuristic-only.")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to initialize LLM client: {e}. Continuing without LLM.")
            self.llm_client = None

        # Instantiate agents
        self.section_identifier = SectionIdentifier(
            llm_client=self.llm_client,
            llm_verify=self.config.get("pipeline", {}).get("llm_verify", True),
        )
        self.summarizer = SummarizerAgent(
            llm_client=self.llm_client,
            max_bullets=self.config.get("pipeline", {}).get("max_bullets", 6),
        )
        self.poster_formatter = PosterFormatter(output_dir=self.output_dir)

    def _load_config(self, path: str) -> dict:
        if not os.path.isfile(path):
            logging.warning(f"Config file not found at {path}. Using defaults.")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as e:
            logging.warning(f"Failed to load config {path}: {e}")
            return {}

    def run_pipeline(
        self,
        pdf_path: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        write_outputs: bool = True,
    ) -> Poster:
        """
        Main entrypoint.
        Returns Poster dataclass instance.
        """
        self.logger.info(f"🚀 Starting pipeline for PDF: {pdf_path}")

        # 1️⃣ Parse PDF into pages & images
        pages, images = parse_pdf_to_pages_and_images(
            pdf_path,
            output_dir=self.output_dir,
            ocr_if_empty=self.config.get("pipeline", {}).get("ocr_if_empty", True),
        )
        self.logger.info(f"Parsed {len(pages)} pages and {len(images)} images")

        # 2️⃣ Clean text
        cleaned_pages = clean_pages_text(
            pages, remove_references=self.config.get("pipeline", {}).get("remove_references", True)
        )
        self.logger.info("Cleaned page texts")

        # 3️⃣ Identify sections
        sections = self.section_identifier.identify_sections_from_pages(cleaned_pages)
        self.logger.info(f"Identified {len(sections)} sections (heuristic + LLM if available)")

        # 4️⃣ Map images
        try:
            sections = self.section_identifier.map_images_to_sections(sections, images)
        except Exception:
            sections = associate_images_to_sections(sections, images)
        self.logger.info("Mapped images to sections")

        # 5️⃣ Summarize sections
        summarized_sections: List[Section] = []
        for sec in sections:
            try:
                summ = self.summarizer.summarize_section(sec)
                sec.metadata["bullets"] = summ.get("bullets", [])
                for img_obj in summ.get("image_refs", []):
                    for real_img in sec.images:
                        if real_img.id == img_obj.get("id") and img_obj.get("caption"):
                            real_img.caption = img_obj["caption"]
                summarized_sections.append(sec)
                self.logger.debug(f"Summarized section: {sec.title}")
            except Exception as e:
                self.logger.exception(f"Failed to summarize section {sec.title}: {e}")
                summarized_sections.append(sec)

        # 6️⃣ Format poster
        poster = self.poster_formatter.format_to_poster(
            title=title or self._guess_title_from_pages(cleaned_pages),
            authors=authors or self._guess_authors_from_pages(cleaned_pages),
            summarized_sections=summarized_sections,
            layout=self.config.get("pipeline", {}).get("layout", {}),
        )
        self.logger.info("Formatted poster dataclass")

        # 7️⃣ Write outputs
        if write_outputs:
            try:
                json_path = self.poster_formatter.poster_to_json(poster, os.path.join(self.output_dir, "poster.json"))
                md_path = self.poster_formatter.write_markdown_file(
                    poster, filename=os.path.join(self.output_dir, "poster_preview.md")
                )
                self.logger.info(f"Saved JSON → {json_path}")
                self.logger.info(f"Saved Markdown → {md_path}")

                if self.config.get("pipeline", {}).get("export_pptx", False):
                    try:
                        pptx_path = self.poster_formatter.poster_to_pptx(
                            poster, os.path.join(self.output_dir, "poster.pptx")
                        )
                        self.logger.info(f"Saved PPTX → {pptx_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to write PPTX: {e}")
            except Exception as e:
                self.logger.exception(f"Failed to write outputs: {e}")

        self.logger.info("✅ Pipeline finished successfully.")
        return poster

    def _guess_title_from_pages(self, pages: List[Tuple[int, str]]) -> Optional[str]:
        if not pages:
            return None
        first_text = pages[0][1] if pages else ""
        if not first_text:
            return None
        lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
        joined = "\n".join(lines[:10])
        if "abstract" in joined.lower():
            title_text = re.split(r"(?i)\babstract\b", joined)[0]
        else:
            title_text = "\n".join(lines[:3])
        title = title_text.strip().replace("\n", " ")
        if len(title) > 200:
            title = title[:200] + "..."
        return title or None

    def _guess_authors_from_pages(self, pages: List[Tuple[int, str]]) -> Optional[List[str]]:
        if not pages:
            return None
        first_text = pages[0][1] or ""
        lines = [ln.strip() for ln in first_text.splitlines() if ln.strip()]
        authors = []
        for i, ln in enumerate(lines[:12]):
            if re.search(r"(?i)abstract", ln):
                break
            if i == 0 and len(ln.split()) > 10:
                continue
            if ("," in ln or ("and" in ln and len(ln.split()) < 12)) and len(ln.split()) < 20:
                authors.append(ln)
        return authors or None
