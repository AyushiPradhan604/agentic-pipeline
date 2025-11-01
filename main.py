# main.py
"""
Entry point for agentic_research_pipeline.

Usage:
    python main.py --pdf path/to/paper.pdf --title "Optional Title" --authors "A. Author;B. Author" --config configs/config.yaml
"""

import argparse
import logging
import os
import yaml
from utils.logger import setup_logging

# Pipeline manager import
try:
    from pipeline.pipeline_manager import PipelineManager
except Exception as e:
    raise RuntimeError("Failed to import pipeline manager. Ensure your PYTHONPATH includes the repo root. Error: %s" % e)

logger = setup_logging("main", log_dir="logs", level=logging.INFO)


def load_config(path: str) -> dict:
    if not os.path.isfile(path):
        logger.warning("Config file not found at %s, proceeding with defaults.", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def parse_args():
    p = argparse.ArgumentParser(description="Run the agentic_research_pipeline on a research PDF.")
    p.add_argument("--pdf", required=True, help="Path to the input PDF")
    p.add_argument("--title", default=None, help="Optional poster title")
    p.add_argument("--authors", default=None, help="Optional authors (semicolon-separated)")
    p.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    p.add_argument("--no-write", dest="write", action="store_false", help="If set, do not write outputs to disk")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = load_config(cfg_path)
    pm = PipelineManager(config_path=cfg_path)
    authors = [a.strip() for a in args.authors.split(";")] if args.authors else None

    logger.info("Starting pipeline for %s", args.pdf)
    poster = pm.run_pipeline(pdf_path=args.pdf, title=args.title, authors=authors, write_outputs=args.write)

    out_dir = pm.output_dir
    logger.info("Pipeline completed. Outputs are in: %s", out_dir)
    # print quick summary
    print("Poster Title:", poster.title)
    print("Authors:", poster.authors)
    print("Sections:", [s.title for s in poster.sections])
    print("Outputs:", os.path.join(out_dir, "poster.json"), os.path.join(out_dir, "poster_preview.md"))
    if cfg.get("pipeline", {}).get("export_pptx", False):
        print("PPTX:", os.path.join(out_dir, "poster.pptx"))


if __name__ == "__main__":
    main()
