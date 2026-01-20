from typing import Tuple, Optional, List
from .base import BaseLLMModel
import torch
import threading
import queue
import time
from unsloth import FastLanguageModel


class UnslothModel(BaseLLMModel):
    """Unsloth model implementation with dynamic batching for high-throughput inference."""

    def __init__(
        self,
        model,
        tokenizer,
        model_name: str = "unsloth-model",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        heartbeat_func=None,
        batch_timeout_ms: float = 500,  # 500ms to accumulate requests from parallel episodes
        max_batch_size: int = 128,  # Increased from 32 to maximize GPU utilization
    ):
        super().__init__(model_name, max_tokens, temperature)
        self.model = model
        self.tokenizer = tokenizer
        self.heartbeat_func = heartbeat_func
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size

        # Batching infrastructure
        self.request_queue = queue.Queue()
        self.stop_batching = False
        self._inference_ready = False

        # Start background batcher thread
        self.batcher_thread = threading.Thread(target=self._batcher_loop, daemon=True)
        self.batcher_thread.start()

    def _ensure_inference_mode(self):
        """Ensure model is in inference mode (call once before batching starts)."""
        if not self._inference_ready:
            FastLanguageModel.for_inference(self.model)
            self._inference_ready = True

    def _batcher_loop(self):
        """Background thread that accumulates requests and processes them in batches."""
        self._ensure_inference_mode()

        while not self.stop_batching:
            batch = []
            batch_start_time = time.time()
            timeout_seconds = self.batch_timeout_ms / 1000.0

            # Accumulate requests until timeout or max batch size
            while True:
                time_remaining = timeout_seconds - (time.time() - batch_start_time)
                if time_remaining <= 0 and len(batch) > 0:
                    break
                if len(batch) >= self.max_batch_size:
                    break

                try:
                    # Wait for requests with timeout
                    request = self.request_queue.get(timeout=max(0.001, time_remaining))
                    batch.append(request)
                except queue.Empty:
                    if len(batch) > 0:
                        break
                    continue

            if len(batch) == 0:
                continue

            # Process the batch
            self._process_batch(batch)

    def _process_batch(self, batch: List[Tuple]):
        """Process a batch of requests."""
        prompts = [req[0] for req in batch]
        temperatures = [req[1] for req in batch]
        events = [req[2] for req in batch]
        result_containers = [req[3] for req in batch]

        try:
            # Tokenize all prompts
            if hasattr(self.tokenizer, "tokenizer"):
                inputs = self.tokenizer.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to("cuda")
            else:
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to("cuda")

            # Generate for the batch
            # Use the first temperature (assuming all similar for simplicity)
            # Add stop token for '}' to terminate JSON generation early
            stop_token_ids = [self.tokenizer.convert_tokens_to_ids('}')] if hasattr(self.tokenizer, 'convert_tokens_to_ids') else None

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=64,  # Reduced from 256 for speed - JSON responses are short
                temperature=temperatures[0] if temperatures[0] is not None else self.temperature,
                use_cache=True,
                do_sample=True,  # Enable sampling for temperature
                eos_token_id=stop_token_ids if stop_token_ids else self.tokenizer.eos_token_id,
            )

            # Decode results
            input_lens = inputs.attention_mask.sum(dim=1)
            for i, (output, input_len) in enumerate(zip(outputs, input_lens)):
                decoded = self.tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                )

                # Stop at first '}'
                if "}" in decoded:
                    decoded = decoded[: decoded.find("}") + 1]

                # Store result and signal completion
                result_containers[i][0] = decoded
                events[i].set()

        except Exception as e:
            # On error, return empty string to all waiting requests
            for i, (event, container) in enumerate(zip(events, result_containers)):
                container[0] = ""
                event.set()

    def send_msg(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        json_format: bool = False,
    ) -> Tuple[str, bool]:
        if self.heartbeat_func:
            self.heartbeat_func()
        if temperature is None:
            temperature = self.temperature

        # Robustness checks
        s_prompt = system_prompt if system_prompt is not None else ""
        u_prompt = user_prompt if user_prompt is not None else ""
        combined_prompt = f"{s_prompt}\n{u_prompt}"

        # Submit request to batch queue
        result_container = [""]  # Mutable container for result
        event = threading.Event()
        self.request_queue.put((combined_prompt, temperature, event, result_container))

        # Wait for result
        event.wait(timeout=30)  # 30 second timeout
        decoded = result_container[0]

        if json_format:
            return self._extract_json(decoded)

        return decoded, False

    def cleanup(self):
        """Stop the batching thread."""
        self.stop_batching = True
        if self.batcher_thread.is_alive():
            self.batcher_thread.join(timeout=2)
