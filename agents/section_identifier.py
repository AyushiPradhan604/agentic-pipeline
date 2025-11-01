# agents/section_identifier.py
"""
Agent 1 — Section Identifier
- Attempts to split extracted paper text (or page-wise text) into logical sections.
- Hybrid approach:
  1) quick rule-based heading detection (common section headings, all-caps headings, numbered headings)
  2) optional LLM verification/refinement via utils.llm_client.LLMClient (if present)
- Also supports mapping ImageRef objects (from utils/pdf_parser) to nearest section by page number or bbox.
"""
from typing import List, Optional, Tuple
import re
import logging
import json

from . import Section, ImageRef

logger = logging.getLogger(__name__)
COMMON_HEADINGS = [
    "abstract", "introduction", "background", "related work", "literature review",
    "method", "methods", "methodology", "materials and methods", "experiments",
    "results", "discussion", "conclusion", "conclusions", "future work", "acknowledgements",
    "references", "appendix", "supplementary"
]

# try to import an LLM client wrapper; fallback to None
try:
    from utils.llm_client import LLMClient
except Exception:
    LLMClient = None


class SectionIdentifier:
    def __init__(self, llm_client: Optional["LLMClient"] = None, llm_verify: bool = True):
        """
        llm_client: an instance of utils.llm_client.LLMClient (optional)
        llm_verify: whether to use LLM for refining sections (if llm_client provided)
        """
        self.llm_client = llm_client
        self.llm_verify = bool(llm_verify and llm_client is not None)

    def identify_sections_from_pages(self, pages: List[Tuple[int, str]]) -> List[Section]:
        """
        Identify only MAJOR academic sections (Abstract, Introduction, Methods, Experiments,
        Results, Analysis, Ablation, Discussion, Conclusion, etc.).
        Skip non-major sections entirely.
        Merge adjacent 'Method' sections (e.g., 2.1 / 2.2) into one combined Method block.
        """
        page_texts = {p: t for p, t in pages}
        merged = []
        for page_num, text in pages:
            marker = f"\n\n[[PAGE {page_num}]]\n"
            merged.append(marker + (text or ""))
        long_text = "\n".join(merged)

        major_headings = [
            "abstract", "introduction", "background", "related work", "literature review",
            "method", "methods", "methodology", "materials and methods",
            "experiments", "experimental setup", "analysis", "results",
            "discussion", "ablation", "conclusion", "conclusions"
        ]

        # Detect candidate headings
        headings = []
        for m in re.finditer(r"(^[A-Za-z][^\n]{0,120}$)", long_text, flags=re.MULTILINE):
            line = m.group(1).strip()
            if len(line.split()) > 8:
                continue
            norm = re.sub(r"[^A-Za-z ]", " ", line).strip().lower()
            if any(re.fullmatch(rf"(\d+(\.\d+)*)?\s*{h}$", norm) for h in major_headings):
                headings.append({"title": line.strip(), "span": m.span()})

        # Fallback search
        if not headings:
            for h in major_headings:
                for m in re.finditer(rf"(^|\n)\s*{re.escape(h)}\s*(\n|:)", long_text, flags=re.IGNORECASE):
                    headings.append({"title": h.capitalize(), "span": m.span()})

        sections = []
        if not headings:
            logger.info("No major headings found — falling back to page-wise segmentation")
            for page_num, text in pages:
                sections.append(Section(title=f"Page {page_num}", text=text or "", start_page=page_num, end_page=page_num))
            return sections

        headings = sorted(headings, key=lambda x: x["span"][0])
        seen_titles = set()
        positions = [h["span"][0] for h in headings] + [len(long_text) + 1]

        method_buffer = None  # temporarily hold merged method text
        last_method_end = None

        for idx, h in enumerate(headings):
            title = h["title"].strip()
            norm_title = re.sub(r"[^A-Za-z ]", " ", title).strip().lower()

            # Identify the canonical heading type
            base_name = next((mt for mt in major_headings if mt in norm_title), None)
            if not base_name:
                continue

            start = h["span"][1]
            end = positions[idx + 1]
            section_text = long_text[start:end].strip()
            sp = self._find_page_from_text(section_text)
            ep = self._find_page_from_text(long_text[start:end])

            # Merge nearby Method sections
            if base_name.startswith("method"):
                if method_buffer is None:
                    # start a new method section
                    method_buffer = {
                        "title": "Method",
                        "text": section_text,
                        "start_page": sp,
                        "end_page": ep,
                        "start_pos": start,
                        "end_pos": end
                    }
                    last_method_end = end
                else:
                    # check proximity to last method section
                    if (start - last_method_end) < 2000:  # within ~2k chars → merge
                        method_buffer["text"] += "\n\n" + section_text
                        method_buffer["end_page"] = ep
                        method_buffer["end_pos"] = end
                        last_method_end = end
                    else:
                        # far apart, finalize previous buffer
                        sections.append(Section(
                            title=method_buffer["title"],
                            text=method_buffer["text"],
                            start_page=method_buffer["start_page"],
                            end_page=method_buffer["end_page"]
                        ))
                        # start new one
                        method_buffer = {
                            "title": "Method",
                            "text": section_text,
                            "start_page": sp,
                            "end_page": ep,
                            "start_pos": start,
                            "end_pos": end
                        }
                        last_method_end = end
                continue

            # For non-method major sections — skip duplicates
            if base_name in seen_titles:
                continue
            seen_titles.add(base_name)

            sections.append(Section(title=title, text=section_text, start_page=sp, end_page=ep))

        # flush final method buffer if any
        if method_buffer:
            sections.append(Section(
                title=method_buffer["title"],
                text=method_buffer["text"],
                start_page=method_buffer["start_page"],
                end_page=method_buffer["end_page"]
            ))

        # Optional LLM refinement (unchanged)
        if self.llm_verify:
            try:
                sections = self._refine_with_llm(sections)
            except Exception as e:
                logger.warning("LLM refine failed, using heuristic sections: %s", e)

        if not sections:
            logger.info("No recognized major sections found; returning empty result.")
        return sections

    def _find_page_from_text(self, text: str) -> Optional[int]:
        m = re.search(r"\[\[PAGE (\d+)\]\]", text)
        if m:
            return int(m.group(1))
        return None

    def _refine_with_llm(self, sections: List[Section]) -> List[Section]:
        """
        Ask the LLM to verify/merge/split sections and return cleaned list.
        The LLM is prompted to return a JSON array with fields: title, start_page, end_page, text.
        """
        if not self.llm_client:
            return sections

        prompt = self._build_llm_refine_prompt(sections)
        resp = self.llm_client.generate(prompt, max_tokens=1024)
        # Expect the LLM to return strict JSON; try to parse gracefully
        try:
            content = resp.strip()
            # Some LLMs prepend text — find first '{' or '['
            start_idx = min((content.find('[') if content.find('[') != -1 else len(content)),
                            (content.find('{') if content.find('{') != -1 else len(content)))
            json_text = content[start_idx:]
            parsed = json.loads(json_text)
            refined = []
            for obj in parsed:
                refined.append(Section(
                    title=obj.get("title") or "Untitled",
                    text=obj.get("text") or "",
                    start_page=obj.get("start_page"),
                    end_page=obj.get("end_page"),
                    images=[]
                ))
            return refined
        except Exception as e:
            logger.warning("Failed to parse LLM refine output: %s", e)
            return sections

    def _build_llm_refine_prompt(self, sections: List[Section]) -> str:
        """
        Build prompt asking the LLM to clean up heuristically detected sections.
        Output MUST be valid JSON array:
        [
          {"title": "...", "start_page": 1, "end_page": 2, "text": "..."}, ...
        ]
        """
        small_samples = []
        for s in sections:
            small_samples.append({
                "title": s.title,
                "start_page": s.start_page,
                "end_page": s.end_page,
                "text_snippet": (s.text[:800] + "...") if s.text else ""
            })
        prompt = (
            "You are a helper that cleans up section segmentation of an academic paper.\n"
            "Input is a list of candidate sections (title, start_page, end_page, text_snippet).\n"
            "Please return a JSON array of sections with keys: title, start_page, end_page, text.\n"
            "Merge or split if needed; drop non-informative sections like 'References'.\n"
            f"Candidates:\n{json.dumps(small_samples, indent=2)}\n\n"
            "Important: return ONLY valid JSON (an array). Keep text reasonably short (under 3000 chars each).\n"
        )
        return prompt

    def map_images_to_sections(self, sections: List[Section], images: List[ImageRef]) -> List[Section]:
        """
        Map images to sections by page number first; if bbox/page missing, best-effort by nearest start_page.
        """
        if not images:
            return sections

        # Index sections by page ranges
        for img in images:
            placed = False
            for sec in sections:
                if sec.start_page and sec.end_page and img.page:
                    if sec.start_page <= img.page <= (sec.end_page or sec.start_page):
                        sec.images.append(img)
                        placed = True
                        break
                elif sec.start_page and img.page:
                    if sec.start_page == img.page:
                        sec.images.append(img)
                        placed = True
                        break
            if not placed:
                # place in closest section by page distance
                if img.page:
                    best = None
                    best_dist = 9999
                    for sec in sections:
                        sp = sec.start_page or 9999
                        dist = abs(sp - img.page)
                        if dist < best_dist:
                            best_dist = dist
                            best = sec
                    if best:
                        best.images.append(img)
                else:
                    # if no page info, append to the section with largest text length (likely Results/Methods)
                    biggest = max(sections, key=lambda s: len(s.text or ""))
                    biggest.images.append(img)
        return sections


# simple usage example (not executed here):
# from utils.pdf_parser import parse_pdf_to_pages_and_images
# pages, images = parse_pdf_to_pages_and_images("paper.pdf")
# sid = SectionIdentifier(llm_client=my_llm)
# sections = sid.identify_sections_from_pages(pages)
# sections = sid.map_images_to_sections(sections, images)