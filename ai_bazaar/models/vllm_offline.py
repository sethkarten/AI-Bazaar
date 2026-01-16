from typing import Tuple, Optional, List
from .base import BaseLLMModel
import torch


class VLLMOfflineModel(BaseLLMModel):
    """vLLM model implementation using the LLM class directly (no server)."""

    def __init__(
        self, model_path: str, max_tokens: int = 1000, temperature: float = 0.7
    ):
        super().__init__(model_path, max_tokens, temperature)
        from vllm import LLM, SamplingParams

        print(f"Initializing Offline vLLM: {model_path}", flush=True)
        self.llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.4,  # Allow room for Unsloth training
            max_model_len=2048,
            disable_log_requests=True,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_tokens, stop=["}"]
        )

    def send_msg(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        json_format: bool = False,
    ) -> Tuple[str, bool]:
        from vllm import SamplingParams

        combined_prompt = f"{system_prompt}\n{user_prompt}"

        params = self.sampling_params
        if temperature is not None:
            params = SamplingParams(
                temperature=temperature, max_tokens=self.max_tokens, stop=["}"]
            )

        outputs = self.llm.generate([combined_prompt], params, use_tqdm=False)
        decoded = outputs[0].outputs[0].text

        if json_format:
            return self._extract_json(decoded)

        return decoded, False

    def generate_batch(
        self, prompts: List[str], temperature: Optional[float] = None
    ) -> List[str]:
        """High-throughput batch generation."""
        from vllm import SamplingParams

        params = self.sampling_params
        if temperature is not None:
            params = SamplingParams(
                temperature=temperature, max_tokens=self.max_tokens, stop=["}"]
            )

        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]
