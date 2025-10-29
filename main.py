import argparse
import yaml
import os
import sys
import json
from datetime import datetime

# Local imports
from pipeline.pipeline_manager import PipelineManager
from utils.logger import get_logger


# ---------------------------
# Helper: Load YAML config
# ---------------------------
def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------
# Helper: Validate JSON input
# ---------------------------
def validate_json_input(json_path: str) -> dict:
    """Ensures the JSON file exists and is well-formed."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ JSON file not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Invalid JSON file format: {e}")

    if not isinstance(data, dict):
        raise ValueError("❌ JSON root must be an object (dictionary).")

    return data


# ---------------------------
# Main Execution
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Agentic Research Pipeline – From Research Paper (JSON/PDF) to Poster"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input research paper (JSON or PDF)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    logger = get_logger("main")
    logger.info("🚀 Starting Agentic Research Pipeline...")
    logger.info(f"📄 Input file: {args.input}")

    # Load config
    config = load_config(args.config)
    os.makedirs(config["paths"]["output_dir"], exist_ok=True)

    # Initialize pipeline
    try:
        pipeline = PipelineManager(config_path=args.config)

        # Detect file type
        if args.input.lower().endswith(".json"):
            logger.info("🧠 Detected JSON input file.")
            final_output = pipeline.run_pipeline(args.input, input_type="json")
        elif args.input.lower().endswith(".pdf"):
            logger.info("📘 Detected PDF input file.")
            final_output = pipeline.run_pipeline(args.input, input_type="pdf")
        else:
            raise ValueError("❌ Unsupported file type. Please provide a .json or .pdf file.")

        # Save output
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(
            config["paths"]["output_dir"],
            f"poster_output_{timestamp}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        logger.info("✅ Pipeline completed successfully!")
        logger.info(f"📝 Output saved at: {output_path}")

    except Exception as e:
        logger.exception(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
