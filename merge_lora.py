import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_path = "/mnt/c/Users/Cameron/OneDrive - Princeton University/Thesis/AI-Bazaar/Models/Qwen3.5-9B"
lora_path = "/mnt/c/Users/Cameron/OneDrive - Princeton University/Thesis/AI-Bazaar/AI-Bazaar/ai-bazaar-checkpoints/lemon_guardian"
output_path = "/mnt/c/Users/Cameron/OneDrive - Princeton University/Thesis/AI-Bazaar/Models/Qwen3.5-9B-guardian"

# Split layers across GPU (7.5GB) and CPU RAM (13GB) — total ~18GB needed for float16 9B model.
# Leaves headroom for OS and other processes.
max_memory = {0: "7500MiB", "cpu": "13000MiB"}

print("Loading base model across GPU + CPU RAM...")
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    torch_dtype=torch.float16,
    device_map="auto",
    max_memory=max_memory,
)
model = PeftModel.from_pretrained(model, lora_path)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {output_path}...")
model.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(output_path)
print("Done.")