"""
utils/image_handler.py
----------------------
Handles mapping of extracted images to their corresponding sections.
Ensures image placeholders like <<Figure_2.png>> are preserved through the pipeline.
"""

import os
import re
from typing import Dict

class ImageHandler:
    def __init__(self):
        pass

    def map_images_to_sections(self, text: str, image_map: Dict[str, str]) -> Dict[str, list]:
        """
        Links image placeholders to the nearest section based on occurrence.
        Returns {section_title: [image_placeholders]}.
        """
        section_images = {}
        section_titles = re.findall(r'\n([A-Z][A-Za-z\s&/-]{3,30})\n', text)

        for section in section_titles:
            section_images[section] = []
            for placeholder in image_map.keys():
                if placeholder in text:
                    section_images[section].append(placeholder)
        return section_images

    def preserve_placeholders(self, section_text: str, image_placeholders: list) -> str:
        """
        Keeps image placeholders intact during summarization.
        """
        for placeholder in image_placeholders:
            if placeholder not in section_text:
                section_text += f"\n\n{placeholder}"
        return section_text
