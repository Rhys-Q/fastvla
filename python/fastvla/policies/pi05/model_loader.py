import os

from .config import PI05BaseOriginalConfig
from .modeling_pi05 import PI0Pytorch


def instantiate_fastvla_pi05(from_pretrained: bool = False, model_path: str | None = None, device="cuda"):
    config = PI05BaseOriginalConfig()
    policy = PI0Pytorch(config)

    if from_pretrained:
        try:
            print("Loading converted PyTorch weights from HuggingFace Hub (lerobot/pi05_base)...")

            # Download the model from HuggingFace Hub
            import safetensors.torch
            from huggingface_hub import snapshot_download

            # Download the entire repository
            if model_path and os.path.exists(model_path):
                cache_dir = model_path
                print(f"Using cached model from: {cache_dir}")
            else:
                cache_dir = snapshot_download(repo_id="lerobot/pi05_base", repo_type="model")
                print(f"Downloaded model to: {cache_dir}")

            # Try to load safetensors format first
            model_file = os.path.join(cache_dir, "model.safetensors")
            if os.path.exists(model_file):
                state_dict = safetensors.torch.load_file(model_file)
                print(f"Loaded {len(state_dict)} parameters from safetensors")
            else:
                raise FileNotFoundError(f"No safetensors file found in {cache_dir}")

            # Load the state dict into the model
            missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)

            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"    - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"    - {key}")
                    print(f"    ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"    - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"    - {key}")
                    print(f"    ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All pretrained weights loaded successfully!")
            else:
                print("Pretrained weights loaded with some missing/unexpected keys (this may be normal)")

        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")
            print("   Using randomly initialized weights...")
            import traceback

            traceback.print_exc()

    policy.to(device)
    return policy
