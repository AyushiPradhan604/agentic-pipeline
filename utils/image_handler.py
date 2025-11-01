# utils/image_handler.py
"""
Image handling utilities:
- associate_images_to_sections: refined image -> section mapping (page-based + bbox heuristic)
- crop_and_save_image: crop an existing image file by bbox and save as new file
- helper to convert bboxes between PDF coordinates and pixel coordinates if needed
"""

import os
import logging
from typing import List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

try:
    from agents import ImageRef, Section
except Exception:
    # lightweight stand-ins if import fails
    from dataclasses import dataclass, field
    from typing import Dict, Any
    @dataclass
    class ImageRef:
        id: str
        page: int
        bbox: Optional[list] = None
        caption: Optional[str] = None
        path: Optional[str] = None
        metadata: dict = field(default_factory=dict)

    @dataclass
    class Section:
        title: str
        text: str
        start_page: Optional[int] = None
        end_page: Optional[int] = None
        images: List[ImageRef] = field(default_factory=list)
        metadata: dict = field(default_factory=dict)


def associate_images_to_sections(sections: List[Section], images: List[ImageRef]) -> List[Section]:
    """
    Try to place images into the most appropriate section.
    Strategy:
      1. If image.page matches a section start_page..end_page, assign there.
      2. Else, assign to closest section by page distance.
      3. If no page info, attach to the longest section (heuristic).
    The function mutates Section.images and returns sections for convenience.
    """
    if not images:
        return sections
    # ensure sections have start_page/end_page values (if not, compute from metadata)
    for img in images:
        placed = False
        if img.page:
            for sec in sections:
                sp = sec.start_page or None
                ep = sec.end_page or None
                if sp and ep:
                    if sp <= img.page <= ep:
                        sec.images.append(img)
                        placed = True
                        break
                elif sp:
                    if sp == img.page:
                        sec.images.append(img)
                        placed = True
                        break
        if not placed and img.page:
            # place in nearest section by distance to start_page
            best = None
            best_dist = 10**9
            for sec in sections:
                sp = sec.start_page or sec.end_page or None
                if sp:
                    dist = abs(sp - img.page)
                    if dist < best_dist:
                        best_dist = dist
                        best = sec
            if best:
                best.images.append(img)
                placed = True
        if not placed:
            # fallback: place into the longest section's images
            if sections:
                longest = max(sections, key=lambda s: len(s.text or ""))
                longest.images.append(img)
    return sections


def crop_and_save_image(image_ref: ImageRef, crop_bbox: Tuple[int, int, int, int], out_dir: str = "data/outputs/images", suffix: str = "_crop") -> Optional[str]:
    """
    Crop an image file according to pixel bbox (left, upper, right, lower) and save.
    image_ref.path must exist.
    Returns path to new cropped image or None on failure.
    """
    if not image_ref.path or not os.path.isfile(image_ref.path):
        logger.warning("Image path missing or not found: %s", image_ref.path)
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(image_ref.path)
        cropped = img.crop(crop_bbox)
        base, ext = os.path.splitext(os.path.basename(image_ref.path))
        out_name = f"{base}{suffix}{ext}"
        out_path = os.path.join(out_dir, out_name)
        cropped.save(out_path)
        logger.info("Saved cropped image to %s", out_path)
        return out_path
    except Exception as e:
        logger.exception("Failed to crop image %s: %s", image_ref.path, e)
        return None