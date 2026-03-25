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
        batch_timeout_ms: float = 100,  # 100ms - balance between batching and latency for desynchronized episodes
        max_batch_size: int = 128,  # Increased from 32 to maximize GPU utilization
        encoding_tokenizer=None,  # For bypassing Gemma3Processor bug
        device=None,  # Specify device for inference (e.g., "cuda:0")
    ):
        super().__init__(model_name, max_tokens, temperature)
        self.model = model
        self.tokenizer = tokenizer
        self.encoding_tokenizer = encoding_tokenizer if encoding_tokenizer is not None else tokenizer
        self.heartbeat_func = heartbeat_func
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.device = device if device is not None else "cuda"

        # Batching infrastructure
        self.request_queue = queue.Queue()
        self.stop_batching = False
        self._inference_ready = False

        # Start background batcher thread
        self.batcher_thread = threading.Thread(target=self._batcher_loop, daemon=True)
        self.batcher_thread.start()

    def _ensure_inference_mode(self):
        """Ensure model is in inference mode."""
        if not self._inference_ready:
            self.model.eval()
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

        # Log batch size for monitoring
        if len(batch) > 1:
            print(f"[BATCH] Processing batch of {len(batch)} requests", flush=True)

        try:
            # Tokenize all prompts
            if hasattr(self.tokenizer, "tokenizer"):
                inputs = self.tokenizer.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)
            else:
                inputs = self.tokenizer(
                    prompts, return_tensors="pt", padding=True, truncation=True
                ).to(self.device)

            # Generate for the batch
            # Use the first temperature (assuming all similar for simplicity)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=temperatures[0] if temperatures[0] is not None else self.temperature,
                use_cache=True,
                do_sample=True,
            )

            # Decode results
            input_lens = inputs.attention_mask.sum(dim=1)
            for i, (output, input_len) in enumerate(zip(outputs, input_lens)):
                # Debug: log raw token count
                output_tokens = output[input_len:]
                if i == 0 and len(batch) > 1:  # Log first item in batch
                    print(f"[DEBUG] Generated {len(output_tokens)} tokens, first 10: {output_tokens[:10].tolist()}", flush=True)

                # Use encoding_tokenizer to bypass Gemma3Processor bug
                if hasattr(self, 'encoding_tokenizer'):
                    decoded = self.encoding_tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                    )
                else:
                    decoded = self.tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                    )

                # Debug: log raw decoded output before JSON extraction
                if i == 0 and len(batch) > 1:
                    print(f"[DEBUG] Raw decoded (first 200 chars): {decoded[:200]!r}", flush=True)

                # Robust JSON extraction: handle markdown, thinking tags, and mixed output
                # 1. Remove markdown code blocks
                if "```json" in decoded:
                    decoded = decoded.split("```json", 1)[1]
                    if "```" in decoded:
                        decoded = decoded.split("```", 1)[0]
                elif "```" in decoded and "{" in decoded:
                    # Generic code block
                    decoded = decoded.split("```", 1)[1]
                    if "```" in decoded:
                        decoded = decoded.split("```", 1)[0]

                # 2. Remove thinking tags if present
                if "</think>" in decoded:
                    decoded = decoded.split("</think>", 1)[1]

                # 3. Extract complete JSON object (find matching closing brace)
                if "{" in decoded and "}" in decoded:
                    # Find the FIRST complete JSON object (most likely to be the answer)
                    start = decoded.find("{")
                    brace_count = 0
                    for idx in range(start, len(decoded)):
                        if decoded[idx] == "{":
                            brace_count += 1
                        elif decoded[idx] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                decoded = decoded[start:idx + 1]
                                break
                else:
                    # No valid JSON found - log for debugging but don't fail silently
                    if i == 0 and len(batch) > 1:
                        print(f"[WARNING] No JSON braces found in output: {decoded[:100]!r}", flush=True)

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

        # Build prompt using chat template for proper role formatting
        s_prompt = system_prompt if system_prompt is not None else ""
        u_prompt = user_prompt if user_prompt is not None else ""
        messages = [{"role": "system", "content": s_prompt}, {"role": "user", "content": u_prompt}]
        try:
            combined_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback for tokenizers without chat template
            combined_prompt = f"{s_prompt}\n{u_prompt}"

        # Submit request to batch queue
        result_container = [""]  # Mutable container for result
        event = threading.Event()
        self.request_queue.put((combined_prompt, temperature, event, result_container))

        # Wait for result - use longer timeout for large batch inference
        # 64 requests × 1024 tokens each can take 2-3 minutes
        timeout_result = event.wait(timeout=300)  # 5 minute timeout
        if not timeout_result:
            print(f"[WARNING] Inference timeout after 300s", flush=True)
        decoded = result_container[0]

        if json_format:
            return self._extract_json(decoded)

        return decoded, False

    def cleanup(self):
        """Stop the batching thread."""
        self.stop_batching = True
        if self.batcher_thread.is_alive():
            self.batcher_thread.join(timeout=2)
