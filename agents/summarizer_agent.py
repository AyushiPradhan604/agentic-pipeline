# agents/summarizer_agent.py
"""
Agent 2 — Summarizer Agent
- For each Section, produce image-aware bullet point summaries.
- Preferred path: call local LLM (Qwen) using utils.llm_client.LLMClient and request strict JSON:
  {
    "section_title": "...",
    "bullets": ["...", "..."],
    "image_refs": [{"id":"img_1","caption":"..."}]
  }
- Fallback: heuristic summarizer (first sentences + sentences containing numbers/keywords).
"""
from typing import List, Dict, Any, Optional
import logging
import re
import json

from . import Section, ImageRef

logger = logging.getLogger(__name__)

try:
    from utils.llm_client import LLMClient
except Exception:
    LLMClient = None

BULLET_PROMPT_TEMPLATE = """
You are an expert summarization assistant for academic papers.
Input: a section title and the section text (will be provided).
Produce a JSON object with the following schema:
{{
  "section_title": "<cleaned section title>",
  "bullets": ["short bullet 1", "short bullet 2", ...],   // 3-8 concise bullets, each <= 30 words
  "image_refs": [{{"id":"img_1","caption":"short caption or extracted caption"}}, ...]
}}
Notes:
- Focus on key contributions, methods, numeric results, novelty, and any claims.
- Bullets should be concise, factual, and self-contained.
- If the section contains essential numeric results, include them.
- Return ONLY valid JSON.
"""

class SummarizerAgent:
    def __init__(self, llm_client: Optional["LLMClient"] = None, max_bullets: int = 6):
        self.llm_client = llm_client
        self.max_bullets = max_bullets
        self.use_llm = bool(llm_client is not None)

    def summarize_section(self, section: Section) -> Dict[str, Any]:
        """
        Summarize a single Section into bullets, preserving image references.
        Returns a dict with keys: section_title, bullets (list), image_refs (list)
        """
        image_refs = [{"id": img.id, "caption": (img.caption or "")} for img in section.images]

        if self.use_llm:
            prompt = BULLET_PROMPT_TEMPLATE + "\n\nSection title:\n" + section.title + "\n\nSection text:\n" + (section.text[:4000] + "..." if len(section.text) > 4000 else section.text)
            try:
                resp = self.llm_client.generate(prompt, max_tokens=512)
                # parse JSON only
                parsed = self._parse_json_from_text(resp)
                if parsed:
                    # ensure image_refs are included
                    parsed.setdefault("image_refs", image_refs)
                    parsed.setdefault("section_title", section.title)
                    # limit bullets
                    parsed["bullets"] = parsed.get("bullets", [])[: self.max_bullets]
                    return parsed
            except Exception as e:
                logger.warning("LLM summarization failed: %s", e)

        # Fallback heuristic summarization
        bullets = self._heuristic_bullets(section.text, max_bullets=self.max_bullets)
        return {"section_title": section.title, "bullets": bullets, "image_refs": image_refs}

    def _heuristic_bullets(self, text: str, max_bullets: int = 6) -> List[str]:
        # split into sentences; pick first sentence, sentences with numbers, and sentences with keywords
        sents = re.split(r"(?<=[\.\?\!])\s+", (text or "").strip())
        picks = []
        if sents:
            picks.append(self._clean_sentence(sents[0]))
        keywords = ["we propose", "we present", "results", "achieve", "improve", "accuracy", "dataset", "experiment", "show"]
        for s in sents[1:]:
            ls = s.lower()
            if any(k in ls for k in keywords):
                picks.append(self._clean_sentence(s))
            elif re.search(r"\d", s) and len(picks) < max_bullets:
                picks.append(self._clean_sentence(s))
            if len(picks) >= max_bullets:
                break
        # pad if too few
        for s in sents[1:]:
            if len(picks) >= max_bullets:
                break
            sentence = self._clean_sentence(s)
            if sentence not in picks and len(sentence) > 20:
                picks.append(sentence)
        # final trimming
        return [p[:300] for p in picks[:max_bullets]]

    def _clean_sentence(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    def _parse_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        # find first { or [ and try to json.loads
        try:
            start = text.find("{")
            if start == -1:
                start = text.find("[")
            candidate = text[start:]
            return json.loads(candidate)
        except Exception as e:
            logger.debug("Failed to parse JSON from LLM output: %s", e)
            return None
