# pipeline/__init__.py
"""
Pipeline package for agentic_research_pipeline.

Exports:
- PipelineManager: high-level orchestrator
- Workflow: reusable step definitions
"""
from .pipeline_manager import PipelineManager
from .workflow import Workflow

__all__ = ["PipelineManager", "Workflow"]
