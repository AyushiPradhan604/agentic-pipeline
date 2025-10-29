"""
tests/test_pipeline_integration.py
----------------------------------
Integration test for the entire Agentic Research Pipeline.
"""

import os
import pytest
import yaml
from pipeline.pipeline_manager import PipelineManager

@pytest.fixture(scope="module")
def config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def pipeline(config):
    return PipelineManager(config_path="configs/config.yaml")

def test_full_pipeline_run(pipeline):
    sample_pdf_path = "data/samples/sample_research_paper.pdf"
    
    # Ensure the sample file exists for test
    if not os.path.exists(sample_pdf_path):
        os.makedirs("data/samples", exist_ok=True)
        with open(sample_pdf_path, "w") as f:
            f.write("INTRODUCTION\nSample intro\nMETHODOLOGY\nSample method\nRESULTS\nSample result")

    result = pipeline.run_pipeline(sample_pdf_path)
    
    assert isinstance(result, dict)
    assert "poster_structure" in result
    assert len(result["poster_structure"].keys()) > 0
