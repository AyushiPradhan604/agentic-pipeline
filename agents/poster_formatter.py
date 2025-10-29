import os
import yaml
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class PosterFormatter:
    """
    Agent 3 – Combines all summarized sections and formats them
    into a structured, poster-ready representation (JSON or Markdown).
    Uses locally loaded Qwen model for text polishing and structure refinement.
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # ✅ Normalize path for Windows
        self.model_path = os.path.normpath("Qwen1.5-0.5B-Chat").replace("\\", "/")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading local Qwen model from: {self.model_path} ({self.device})")

        # ✅ Load tokenizer and model directly from local directory
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

    def format_for_poster(self, summarized_sections):
        """Combines summaries into a coherent poster-style output."""
        if not summarized_sections:
            raise ValueError("No summarized sections provided to PosterFormatter.")

        combined_text = self._build_structured_text(summarized_sections)
        prompt = self._build_prompt(combined_text)
        formatted_output = self._query_qwen(prompt)

        # ✅ Save output in data/outputs
        output_path = os.path.join("data", "outputs", "final_poster_structure.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"poster_content": formatted_output}, f, indent=4, ensure_ascii=False)

        return formatted_output

    def _build_structured_text(self, summarized_sections):
        """Combines section summaries in a structured order."""
        ordered_sections = self.config.get("sections", [])
        text = ""
        for sec in ordered_sections:
            for s in summarized_sections:
                if s["section"].lower() == sec.lower():
                    text += f"\n### {sec}\n{s['summary']}\n\n"
        if not text:
            text = "\n".join([f"### {s['section']}\n{s['summary']}\n" for s in summarized_sections])
        return text.strip()

    def _build_prompt(self, combined_text):
        """Prompt for Qwen to produce structured poster layout."""
        return f"""
You are a scientific poster layout assistant.
Combine and format the following summarized sections
into a coherent, visually structured poster layout.

Ensure:
- Each section is clearly titled.
- Bullets are concise.
- Image placeholders (like <<Figure_2.png>>) remain exactly where they are.
- Maintain logical flow: Introduction → Motivation → Dataset → Methodology → Results → Ablation → Conclusion.

Input Sections:
\"\"\"{combined_text[:4000]}\"\"\"


Output format:
{{
  "poster_layout": [
    {{
      "section": "<Section Name>",
      "content": [
        "- Bullet 1",
        "- Bullet 2"
      ],
      "images": ["<<Figure_1.png>>"]
    }}
  ]
}}
"""

    def _query_qwen(self, prompt: str):
        """Run local Qwen model for formatting."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                do_sample=False
            )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result


# Example standalone usage
if __name__ == "__main__":
    example_summaries = [
        {"section": "Introduction", "summary": "- Goal of research\n- Motivation\n", "images": []},
        {"section": "Methodology", "summary": "- Used CNN model (<<Figure_1.png>>)\n- Data split 80/20\n", "images": ["<<Figure_1.png>>"]}
    ]

    formatter = PosterFormatter()
    formatted_output = formatter.format_for_poster(example_summaries)
    print("\n[Poster-Ready Layout]")
    print(formatted_output)
