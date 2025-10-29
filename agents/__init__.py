"""
agents/__init__.py
------------------
Initializes and exposes the three main agents for the Agentic Research Pipeline.
Now configured to load Qwen models from the local path.
"""

import os
from .section_identifier import SectionIdentifier
from .summarizer_agent import SummarizerAgent
from .poster_formatter import PosterFormatter

__all__ = [
    "SectionIdentifier",
    "SummarizerAgent",
    "PosterFormatter",
    "load_all_agents",
]


def load_all_agents(config=None):
    """
    Initializes all agents with the given configuration (if any).
    Each agent loads the Qwen model from the local directory:
        agentic_research_pipeline/Qwen1.5-0.5B-Chat/

    Args:
        config (dict, optional): Configuration dictionary (optional since paths are fixed).

    Returns:
        tuple: (section_identifier, summarizer, poster_formatter)
    """
    base_path = os.path.join(os.path.dirname(__file__), "..")
    local_model_path = os.path.join(base_path, "Qwen1.5-0.5B-Chat")

    print(f"[INFO] Initializing all agents using local Qwen model at: {local_model_path}")

    section_identifier = SectionIdentifier(config_path="configs/config.yaml")
    summarizer = SummarizerAgent(config_path="configs/config.yaml")
    poster_formatter = PosterFormatter(config_path="configs/config.yaml")

    return section_identifier, summarizer, poster_formatter
