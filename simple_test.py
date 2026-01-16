import torch
import time

print("Starting local test...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")

for i in range(5):
    print(f"Step {i}...")
    time.sleep(2)

print("Local test completed successfully!")
