"""
tests/test_agent2_summarizer_agent.py
-------------------------------------
Unit tests for the SummarizerAgent.
"""

import pytest
from agents.summarizer_agent import SectionSummarizerAgent

import yaml

@pytest.fixture(scope="module")
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def agent(config):
    return SectionSummarizerAgent(config)

def test_summarization_preserves_placeholders(agent):
    section_name = "Methodology"
    section_text = """
    We used a CNN-based architecture as shown in <<Figure_1.png>> for training.
    """

    summary = agent.summarize(section_name, section_text)

    assert isinstance(summary, dict)
    assert "bullets" in summary
    assert any("Figure_1" in b for b in summary["bullets"])
    assert "images" in summary
    assert "<<Figure_1.png>>" in summary["images"]
