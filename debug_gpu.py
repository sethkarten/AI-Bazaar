import torch
from unsloth import FastLanguageModel
import time
import os

model_name = "./models/gemma-3-4b-it-bnb-4bit"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Dummy training step
batch_size = 32  # Stress test
seq_len = 512
print(
    f"Starting dummy backward pass with batch_size={batch_size}, seq_len={seq_len}..."
)

# Synthetic data
input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to("cuda")
labels = input_ids.clone()

# Forward pass
outputs = model(input_ids, labels=labels)
loss = outputs.loss

# Backward pass
loss.backward()
optimizer.step()

print("Backward pass completed.")
print(f"Current VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Max VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

time.sleep(10)  # Keep alive to check nvidia-smi
