"""
agentic_research_pipeline/pipeline/workflow.py
----------------------------------------------
Defines the high-level workflow logic for running the Agentic Research Pipeline.
"""

from pipeline.pipeline_manager import PipelineManager
from utils.logger import get_logger

class PipelineWorkflow:
    def __init__(self, config_path: str = "agentic_research_pipeline/configs/config.yaml"):
        self.logger = get_logger(__name__)
        self.pipeline_manager = PipelineManager(config_path)
        self.logger.info("PipelineWorkflow initialized.")

    def run_single_paper(self, pdf_path: str):
        """
        Run the pipeline for a single research paper.
        """
        self.logger.info(f"Starting workflow for {pdf_path}")
        result = self.pipeline_manager.run_pipeline(pdf_path)
        self.logger.info("Workflow completed successfully.")
        return result

    def run_batch(self, pdf_list):
        """
        Run the pipeline on a list of research papers sequentially.
        """
        results = []
        for pdf in pdf_list:
            self.logger.info(f"Processing batch item: {pdf}")
            output = self.pipeline_manager.run_pipeline(pdf)
            results.append(output)
        return results
