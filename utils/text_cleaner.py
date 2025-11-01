# utils/text_cleaner.py
"""
Text cleaning helpers:
- detect and remove repeated headers/footers appearing on many pages
- remove references section (truncate)
- remove inline citation noise like [1], (Smith et al., 2020)
- normalize whitespace and fix broken line breaks
"""

from typing import List, Tuple
import re
import logging
logger = logging.getLogger(__name__)

def merge_pages_text(pages: List[Tuple[int, str]], join_with="\n\n") -> str:
    """Concatenate page-wise text into a single document string, preserving page markers."""
    parts = []
    for pnum, text in pages:
        parts.append(f"[[PAGE {pnum}]]\n" + (text or ""))
    return join_with.join(parts)


def clean_pages_text(pages: List[Tuple[int, str]], remove_references: bool = True, min_repeat_pct: float = 0.35) -> List[Tuple[int, str]]:
    """
    Clean page-wise texts and return new pages list.
    Steps:
      - detect repeated header/footer lines occurring on >= min_repeat_pct of pages and remove them
      - optionally truncate at 'References' or 'Bibliography' marker
      - remove inline citation markers and normalize whitespace
    """
    # split pages into lines and detect repeated lines
    page_lines = []
    for pnum, text in pages:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        page_lines.append((pnum, lines))

    # gather line frequencies (use first/last 6 lines as headers/footers candidates)
    candidate_lines = []
    for pnum, lines in page_lines:
        head = lines[:6]
        foot = lines[-6:] if len(lines) >= 6 else lines
        candidate_lines.extend(head)
        candidate_lines.extend(foot)

    freq = {}
    total_pages = max(1, len(pages))
    for ln in set(candidate_lines):
        freq[ln] = sum(1 for _, lines in page_lines if ln in lines)

    repeated = set(ln for ln, count in freq.items() if count / total_pages >= min_repeat_pct)
    logger.debug("Detected %d repeated header/footer lines", len(repeated))

    cleaned_pages = []
    truncated = False
    for pnum, lines in page_lines:
        # remove repeated lines
        new_lines = [ln for ln in lines if ln not in repeated]
        text = "\n".join(new_lines)

        # remove page markers like "Page 1 of 10"
        text = re.sub(r"page\s*\d+\s*(of\s*\d+)?", "", text, flags=re.IGNORECASE)

        # basic citation cleanup: [1], [12, 13]
        text = re.sub(r"\[\s*\d+(?:[\s,]*\d+)*\s*\]", "", text)

        # remove (Smith et al., 2020) style citations — apply to the `text` string
        text = re.sub(r"\((?:[A-Z][A-Za-z\-\s]+?,\s*\d{4})(?:;[^\)]*)*\)", "", text)

        # replace "(et al.)" style safely on the `text` string
        text = re.sub(r"\s*\(\s*et al\.\s*\)", " et al.", text, flags=re.IGNORECASE)

        # Normalize dashes and weird unicode
        text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\xa0", " ")

        # Remove common OCR artifacts like long hyphenation at line ends
        text = re.sub(r"-\s*\n\s*", "", text)
        # collapse multi-space and fix newlines
        text = re.sub(r"\s+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Optionally truncate at References/Bibliography/ACKNOWLEDGEMENTS depending on flags
        if remove_references and not truncated:
            # naive detection in the page text
            match = re.search(r"^\s*(references|bibliography|acknowledg(e)?ments)\b", text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                # keep only text before the header
                parts = re.split(r"^\s*(references|bibliography|acknowledg(e)?ments)\b", text, flags=re.IGNORECASE | re.MULTILINE)
                if parts:
                    text = parts[0].strip()
                truncated = True

        text = text.strip()
        cleaned_pages.append((pnum, text))

    return cleaned_pages
