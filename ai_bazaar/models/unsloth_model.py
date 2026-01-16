from typing import Tuple, Optional
from .base import BaseLLMModel
import torch


class UnslothModel(BaseLLMModel):
    """Unsloth model implementation for in-process inference during training."""

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str = "unsloth-model",
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ):
        super().__init__(model_name, max_tokens, temperature)
        self.model = model
        self.tokenizer = tokenizer

    def send_msg(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        json_format: bool = False,
    ) -> Tuple[str, bool]:
        if temperature is None:
            temperature = self.temperature

        # Robustness checks
        s_prompt = system_prompt if system_prompt is not None else ""
        u_prompt = user_prompt if user_prompt is not None else ""
        combined_prompt = f"{s_prompt}\n{u_prompt}"

        # Ensure model is in inference mode
        from unsloth import FastLanguageModel

        if not getattr(self.model, "_is_inference", False):
            FastLanguageModel.for_inference(self.model)
            self.model._is_inference = True

        try:
            inputs = self.tokenizer(combined_prompt, return_tensors="pt").to("cuda")
        except Exception as e:
            self.logger.error(
                f"Tokenizer failed: {e}. Prompt type: {type(combined_prompt)}"
            )
            return "", False

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=temperature,
            use_cache=True,
        )

        # Decode only the new tokens
        input_len = inputs.input_ids.shape[1]
        decoded = self.tokenizer.batch_decode(
            outputs[:, input_len:], skip_special_tokens=True
        )[0]

        if json_format:
            return self._extract_json(decoded)

        return decoded, False
