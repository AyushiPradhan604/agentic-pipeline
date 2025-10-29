"""
tests/test_agent1_section_identifier.py
---------------------------------------
Unit tests for the SectionIdentifierAgent.
"""

import pytest
from agents.section_identifier import SectionIdentifierAgent
import yaml

@pytest.fixture(scope="module")
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def agent(config):
    return SectionIdentifierAgent(config)

def test_section_identification(agent):
    sample_text = """
    INTRODUCTION
    This research aims to improve energy efficiency.

    METHODOLOGY
    We used a CNN-based architecture.

    RESULTS
    Our model achieves 90% accuracy.
    """

    result = agent.identify_sections(sample_text)
    assert isinstance(result, dict)
    assert "Introduction" in [k.capitalize() for k in result.keys()]
    assert len(result) >= 2
