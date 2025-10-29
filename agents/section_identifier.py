import os
import re
import yaml
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SectionIdentifier:
    """
    Agent 1 – Identifies and extracts standard sections from a research paper.
    Uses a locally loaded Qwen model for semantic section detection.
    Now supports JSON input instead of PDF.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        # ----------------------------
        # Load configuration
        # ----------------------------
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # ✅ Normalize model path and choose device
        self.model_path = "Qwen1.5-0.5B-Chat".replace("\\", "/")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading local Qwen model from: {self.model_path} ({self.device})")

        # ✅ Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # ✅ Load section names
        self.target_sections = self.config.get("sections", [])
        if not self.target_sections:
            print("[WARN] 'sections' not found in config. Using default set.")
            self.target_sections = [
                "Abstract", "Introduction", "Motivation", "Related Work",
                "Dataset", "Methodology", "Results", "Ablation", "Conclusion"
            ]

    # -------------------------------------------------------------------------
    def process(self, json_path: str):
        """
        Extracts text from a JSON file (instead of a PDF),
        identifies major sections, and returns them.
        Called directly by PipelineManager.
        """
        print(f"[INFO] Reading research paper from JSON: {json_path}")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ JSON file not found at: {json_path}")

        # Load JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Expect structure like:
        # {"sections": [{"section": "Introduction", "text": "..."}, ...]}
        if isinstance(data, dict) and "sections" in data:
            sections_data = data["sections"]
        elif isinstance(data, list):
            sections_data = data
        else:
            raise ValueError("❌ JSON must contain a list of sections or a 'sections' key.")

        print(f"[INFO] Loaded {len(sections_data)} sections from JSON.")
        sections_output = []

        # Process each section text
        for sec in sections_data:
            section_name = sec.get("section", "Unknown")
            section_text = sec.get("text", "")
            if not section_text.strip():
                print(f"[WARN] Empty text for section '{section_name}', skipping.")
                continue

            prompt = self._build_prompt(section_name, section_text)
            result = self._query_qwen(prompt)
            if result:
                sections_output.append(result)

        print(f"[INFO] Successfully identified {len(sections_output)} cleaned sections.")
        return sections_output

    # -------------------------------------------------------------------------
    def _build_prompt(self, section_name: str, section_text: str):
        """
        Builds a focused instruction prompt for Qwen.
        """
        return f"""
You are a research paper structure analyzer.
From the following text, identify whether it belongs to one of these sections:
{', '.join(self.target_sections)}.

If it does, clean the section content (remove noise like references, equations, or extra symbols),
and return only the refined text for that section.

Section Candidate: {section_name}
Text:
\"\"\"{section_text[:2000]}\"\"\"

Output format (JSON):
{{"section": "<best_matching_section_name>", "content": "<cleaned_text>"}}
"""

    # -------------------------------------------------------------------------
    def _query_qwen(self, prompt: str):
        """
        Runs inference with local Qwen model and parses its JSON output safely.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=False
            )

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # ✅ Extract JSON-like output safely
        match = re.search(r"\{.*\}", result, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict) and "section" in parsed and "content" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass

        # fallback
        return {"section": "Unknown", "content": result.strip()}


# ✅ Example standalone usage
if __name__ == "__main__":
    # Example JSON file structure
    example_json = {
        "sections": [
            {"section": "Introduction", "text": "This paper introduces a novel algorithm for EV navigation."},
            {"section": "Methodology", "text": "The method leverages a CNN architecture with reinforcement learning."}
        ]
    }

    # Save temp file for testing
    test_path = "sample_paper.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(example_json, f, indent=4)

    agent = SectionIdentifier()
    sections = agent.process(test_path)

    print("\n[Extracted Sections]")
    for sec in sections:
        print(f"--- {sec['section'].upper()} ---\n{sec['content'][:300]}...\n")
