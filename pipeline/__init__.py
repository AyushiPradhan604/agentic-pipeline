"""
pipeline/__init__.py
--------------------
Initializes the pipeline module that connects and manages all agents.
"""

from .pipeline_manager import PipelineManager
from .workflow import PipelineWorkflow

__all__ = ["PipelineManager", "PipelineWorkflow"]
