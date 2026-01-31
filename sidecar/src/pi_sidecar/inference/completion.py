"""Text completion using HuggingFace transformers."""

from typing import Optional, List
import torch


class CompletionEngine:
    """Generate text completions using HuggingFace models."""

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def complete(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        if self.model is None:
            self._load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode and remove the prompt
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated[len(prompt):].strip()
        
        # Handle stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                if stop in completion:
                    completion = completion[:completion.index(stop)]
        
        return completion

    def load_model(self, model_path: str) -> None:
        """Load a model from a local path."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()


# Singleton instance
_completion_engine: Optional[CompletionEngine] = None


def get_completion_engine() -> CompletionEngine:
    """Get the singleton completion engine."""
    global _completion_engine
    if _completion_engine is None:
        _completion_engine = CompletionEngine()
    return _completion_engine
