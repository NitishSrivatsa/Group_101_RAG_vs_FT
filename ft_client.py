import os, time
from typing import Optional
from huggingface_hub import InferenceClient

class FTClient:
    """
    Thin client over HF Inference API. Works with your fine-tuned model repo (seq2seq like flan-t5).
    Set env vars in Streamlit Cloud:
      - HUGGINGFACEHUB_API_TOKEN
      - FT_MODEL_REPO (your fine-tuned model id, e.g., 'nitish/finqa-flan-t5')
      - GEN_MODEL_REPO (optional fallback, default 'google/flan-t5-base')
    """
    def __init__(self):
        token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            self.client = None
            self.repo = None
            return
        self.repo = os.environ.get("FT_MODEL_REPO", os.environ.get("GEN_MODEL_REPO", "google/flan-t5-base"))
        self.client = InferenceClient(model=self.repo, token=token)

    def available(self) -> bool:
        return self.client is not None

    def generate_text(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        if not self.client:
            raise RuntimeError("No HF token configured")
        # For seq2seq, we pass the prompt as input text
        out = self.client.text_generation(prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=temperature>0.01)
        return out.strip()
