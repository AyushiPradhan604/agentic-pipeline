"""
utils/text_cleaner.py
---------------------
Provides functions for cleaning and normalizing text extracted from PDFs.
"""

import re

class TextCleaner:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """
        Clean research paper text by removing unwanted artifacts.
        """
        # Remove multiple newlines
        text = re.sub(r'\n+', '\n', text)
        # Remove page numbers
        text = re.sub(r'\bPage\s*\d+\b', '', text)
        # Remove reference-style [1], [2], etc.
        text = re.sub(r'\[\d+\]', '', text)
        # Remove excessive spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def split_into_sections(self, text: str) -> list:
        """
        Roughly splits text into potential section blocks (for SectionIdentifierAgent).
        """
        pattern = r'(?<=\n)(?P<section>[A-Z][A-Za-z\s&/-]{3,30})(?=\n)'
        parts = re.split(pattern, text)
        return [p.strip() for p in parts if len(p.strip()) > 0]
