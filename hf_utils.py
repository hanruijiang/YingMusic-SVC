import os
from huggingface_hub import hf_hub_download


def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin"):
    os.makedirs("./checkpoints", exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id, filename=model_filename, cache_dir="./checkpoints"
    )
