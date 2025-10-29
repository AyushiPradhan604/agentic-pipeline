"""
pipeline/pipeline_manager.py
----------------------------
Handles orchestration of all three agents sequentially:
1. SectionIdentifier
2. SummarizerAgent
3. PosterFormatter
Supports both PDF and JSON input.
"""

import os
import yaml
import json
from utils.logger import get_logger

# Import the three agents
from agents.section_identifier import SectionIdentifier
from agents.summarizer_agent import SummarizerAgent
from agents.poster_formatter import PosterFormatter


class PipelineManager:
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize the pipeline with configuration and agent setup.
        """
        self.logger = get_logger(__name__)
        self.logger.info("Initializing PipelineManager...")

        # Load YAML configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"❌ Config file not found at {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # Initialize all agents
        self.section_identifier = SectionIdentifier(config_path)
        self.summarizer = SummarizerAgent(config_path)
        self.poster_formatter = PosterFormatter(config_path)

        self.logger.info("✅ All agents initialized successfully.")

    # -------------------------------------------------------------------------
    def run_pipeline(self, input_path: str, input_type: str = "json", output_dir: str = "data/outputs/"):
        """
        Runs the complete 3-step Agentic AI pipeline.
        Supports both JSON and PDF input formats.
        """
        self.logger.info(f"🚀 Running pipeline for file: {input_path}")

        # ----------------------------
        # STEP 1: Load or extract sections
        # ----------------------------
        self.logger.info("🔹 Step 1: Identifying sections...")

        if input_type.lower() == "json":
            self.logger.info("📘 Reading sections directly from JSON input...")
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sections_list = []

            # 1️⃣ Extract abstract
            if "pdf_parse" in data and "abstract" in data["pdf_parse"]:
                for abs_item in data["pdf_parse"]["abstract"]:
                    text = abs_item.get("text", "").strip()
                    if text:
                        sections_list.append({"section": "Abstract", "text": text})

            # 2️⃣ Extract body_text sections
            if "pdf_parse" in data and "body_text" in data["pdf_parse"]:
                for item in data["pdf_parse"]["body_text"]:
                    section_name = item.get("section", "Body")
                    text = item.get("text", "").strip()
                    if text:
                        sections_list.append({"section": section_name, "text": text})

            # 3️⃣ Fallback / Validation
            if not sections_list:
                raise ValueError("❌ JSON does not contain valid 'pdf_parse' → 'abstract' or 'body_text' sections.")

            section_names = [s.get("section", f"Section_{i}") for i, s in enumerate(sections_list)]

        elif input_type.lower() == "pdf":
            self.logger.info("📄 Extracting text from PDF using SectionIdentifier...")
            sections = self.section_identifier.process(input_path)

            # Normalize to list format
            if isinstance(sections, dict):
                section_names = list(sections.keys())
                sections_list = [{"section": k, "text": v} for k, v in sections.items()]
            elif isinstance(sections, list):
                section_names = [s.get("section", f"Section_{i}") for i, s in enumerate(sections)]
                sections_list = sections
            else:
                raise TypeError("❌ Unexpected type returned from SectionIdentifier.process().")
        else:
            raise ValueError("❌ Unsupported input type. Use 'json' or 'pdf'.")

        self.logger.info(f"✅ Extracted {len(section_names)} sections: {section_names}")

        # ----------------------------
        # STEP 2: Summarization
        # ----------------------------
        self.logger.info("🔹 Step 2: Summarizing sections...")
        summarized_sections = []

        for s in sections_list:
            section_name = s.get("section", "Unknown")
            section_text = s.get("text", "")
            if not section_text.strip():
                self.logger.warning(f"⚠️ Empty text for section '{section_name}', skipping summarization.")
                continue
            self.logger.info(f"   → Summarizing section: {section_name}")
            summary_data = self.summarizer.summarize_section(section_name, section_text)
            summarized_sections.append(summary_data)

        self.logger.info("✅ All sections summarized successfully.")

        # ----------------------------
        # STEP 3: Poster Formatting
        # ----------------------------
        self.logger.info("🔹 Step 3: Formatting final poster output...")
        poster_output = self.poster_formatter.format_for_poster(summarized_sections)

        # ----------------------------
        # STEP 4: Save Output
        # ----------------------------
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "poster_output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(poster_output, f, indent=4, ensure_ascii=False)

        self.logger.info("🎉 Pipeline completed successfully.")
        self.logger.info(f"📝 Output saved at: {output_path}")

        return poster_output
