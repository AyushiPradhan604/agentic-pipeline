import os
import torch
import re
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


class SummarizerAgent:
    """
    Agent 2 – Summarizes each section into bullet points,
    keeping all image placeholders (like <<Figure_2.png>>) intact.
    Uses locally loaded Qwen model for summarization.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # ✅ Load local Qwen model path
        self.model_path = "Qwen1.5-0.5B-Chat".replace("\\", "/")

        # ✅ Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[INFO] Loading local Qwen model from: {self.model_path} ({self.device})")

        # ✅ Load model + tokenizer locally
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

    # -------------------------------------------------------------------------
    def summarize_section(self, section_name: str, section_text: str):
        """Summarizes a given section into bullet points while preserving image tags."""
        image_refs = re.findall(r"<<[^<>]+>>", section_text)
        prompt = self._build_prompt(section_name, section_text, image_refs)

        summary = self._query_qwen(prompt)

        # ✅ Ensure summary is never None
        if not summary or not isinstance(summary, str):
            summary = "(No summary generated — possibly empty input or model timeout.)"

        return {
            "section": section_name.strip() if section_name else "Unknown",
            "summary": summary.strip(),
            "images": image_refs
        }

    # -------------------------------------------------------------------------
    def _build_prompt(self, section_name: str, section_text: str, image_refs):
        """Creates an instruction for the model."""
        image_hint = "\n".join(image_refs) if image_refs else "None"

        return f"""
You are an expert summarizer for academic research papers.

Summarize the following section: **{section_name}**
into concise bullet points (3–7 bullets max).
Keep every image placeholder (like <<Figure_1.png>>) exactly as it appears.
Do NOT remove or modify them.

Section Text:
\"\"\"{section_text[:2500]}\"\"\"  # limit text length for efficiency

Detected image placeholders:
{image_hint}

Output format:
- Bullet 1
- Bullet 2
- ...
(Keep image placeholders intact)
"""

    # -------------------------------------------------------------------------
    def _query_qwen(self, prompt: str):
        """Runs local inference using Qwen model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.4,
                    do_sample=False
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # ✅ Clean common artifacts
            result = re.sub(r"<\|im_end\|>.*", "", result)
            result = result.strip()

            # ✅ Return clean text or None if empty
            return result if result else None

        except Exception as e:
            print(f"[ERROR] Qwen generation failed: {e}")
            return None
