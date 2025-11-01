import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_path: str = None, backend: str = "qwen-local", device: str = "cuda" if False else "cpu"):
        """
        Simple local LLM client for Qwen or other HuggingFace models.
        """
        self.backend = backend
        self.model_path = model_path
        self.device = device

        if backend == "qwen-local" and model_path:
            logger.info(f"🧠 Loading local Qwen model from {model_path} ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
            self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=-1)
            logger.info("✅ Local Qwen model loaded successfully.")
        else:
            logger.warning(f"⚠️ LLMClient initialized with unsupported backend: {backend}")
            self.pipe = None

    def generate(self, prompt: str, max_new_tokens: int = 400) -> str:
        """
        Generate text response from the LLM.
        """
        if not self.pipe:
            logger.warning("LLMClient.generate called without active model pipeline.")
            return ""

        try:
            outputs = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_p=0.9)
            return outputs[0]["generated_text"]
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
