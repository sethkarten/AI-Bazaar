"""Test batching throughput directly."""
import time
import threading
from ai_bazaar.models.unsloth_model import UnslothModel
from unsloth import FastLanguageModel

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
print("Model loaded!")

# Create UnslothModel with batching
llm = UnslothModel(model, tokenizer, batch_timeout_ms=80, max_batch_size=32)

# Test: Send 100 concurrent requests
NUM_REQUESTS = 100
results = []
errors = []

def send_request(idx):
    try:
        start = time.time()
        response, _ = llm.send_msg(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'hello' in JSON format: {\"message\": \"hello\"}",
            json_format=False
        )
        elapsed = time.time() - start
        results.append((idx, elapsed, len(response)))
        print(f"Request {idx}: {elapsed:.2f}s, {len(response)} chars", flush=True)
    except Exception as e:
        errors.append((idx, str(e)))
        print(f"Request {idx} FAILED: {e}", flush=True)

print(f"\nSending {NUM_REQUESTS} concurrent requests...")
start_time = time.time()

threads = []
for i in range(NUM_REQUESTS):
    t = threading.Thread(target=send_request, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

total_time = time.time() - start_time

print(f"\n=== RESULTS ===")
print(f"Total requests: {NUM_REQUESTS}")
print(f"Successful: {len(results)}")
print(f"Failed: {len(errors)}")
print(f"Total time: {total_time:.2f}s")
print(f"Throughput: {NUM_REQUESTS/total_time:.2f} requests/s")
if results:
    avg_latency = sum(r[1] for r in results) / len(results)
    print(f"Avg latency: {avg_latency:.2f}s")
