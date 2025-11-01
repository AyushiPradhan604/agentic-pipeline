# agents/__init__.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

__all__ = ["Section", "ImageRef", "Poster"]

@dataclass
class ImageRef:
    """
    Represents an extracted image from the paper.
    - id: unique id (e.g., img_1.png)
    - page: source page number (1-indexed)
    - bbox: optional bounding box (x0, y0, x1, y1) in PDF units if available
    - caption: caption text if available
    - path: path to saved image file
    """
    id: str
    page: int
    bbox: Optional[List[float]] = None
    caption: Optional[str] = None
    path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Section:
    """
    Represents a logical section extracted from paper text.
    - title: section title (e.g., "Introduction")
    - start_page: page where section starts (optional)
    - end_page: page where section ends (optional)
    - text: full section text
    - images: list of ImageRef instances mapped to this section
    - metadata: any additional signals (confidence, heuristics used)
    """
    title: str
    text: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    images: List[ImageRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Poster:
    """
    Structured poster ready output.
    - title, authors: top-level metadata
    - sections: list of Section objects with summarized bullet points in metadata (or separate field)
    - layout: hints for layout engine
    """
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    sections: List[Section] = field(default_factory=list)
    layout: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
