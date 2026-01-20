#!/usr/bin/env python3
"""Debug script to test Gemma3 tokenizer behavior."""
import sys
import os
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from unsloth import FastLanguageModel

# Load Gemma3 model
model_path = "/media/milkkarten/data/AI-Bazaar/models/gemma-3-4b-it-bnb-4bit"
print(f"Loading model from {model_path}...")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    print(f"✅ Model loaded successfully")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")
    print(f"Has eos_token: {tokenizer.eos_token}")
    print(f"Has pad_token: {tokenizer.pad_token}")
    print(f"Has apply_chat_template: {hasattr(tokenizer, 'apply_chat_template')}")

    # Test apply_chat_template
    msg = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]

    print("\n" + "="*80)
    print("Testing apply_chat_template...")
    try:
        result = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        print(f"Result type: {type(result)}")
        print(f"Result value: {result[:100] if result else 'None'}")
    except Exception as e:
        print(f"❌ apply_chat_template failed: {e}")

    # Test tokenizer call with list of strings
    print("\n" + "="*80)
    print("Testing tokenizer with list of strings...")
    test_texts = ["Hello world", "How are you?"]
    try:
        enc = tokenizer(test_texts, return_tensors="pt", padding=True, truncation=True)
        print(f"✅ Tokenization successful")
        print(f"input_ids shape: {enc.input_ids.shape}")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()

    # Test with single string
    print("\n" + "="*80)
    print("Testing tokenizer with single string...")
    try:
        enc = tokenizer("Hello world", return_tensors="pt")
        print(f"✅ Tokenization successful")
    except Exception as e:
        print(f"❌ Tokenization failed: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
