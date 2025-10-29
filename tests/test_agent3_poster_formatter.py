"""
tests/test_agent3_poster_formatter.py
-------------------------------------
Unit tests for the PosterFormatterAgent.
"""

import pytest
from agents.poster_formatter import PosterFormatterAgent
import yaml

@pytest.fixture(scope="module")
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def agent(config):
    return PosterFormatterAgent(config)

def test_poster_formatting(agent):
    summarized_data = {
        "Introduction": {
            "bullets": ["- This is intro"],
            "images": ["<<Figure_1.png>>"]
        },
        "Results": {
            "bullets": ["- Model accuracy improved by 10%"],
            "images": []
        }
    }

    result = agent.format_for_poster(summarized_data)
    assert "poster_structure" in result
    assert "Introduction" in result["poster_structure"]
    assert "Results" in result["poster_structure"]
