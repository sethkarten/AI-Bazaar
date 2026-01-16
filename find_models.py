from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(author="unsloth", search="Qwen")
for m in models:
    if "7B" in m.modelId:
        print(m.modelId)
models = api.list_models(author="unsloth", search="OLMo")
for m in models:
    print(m.modelId)
models = api.list_models(author="unsloth", search="Ministral")
for m in models:
    print(m.modelId)
