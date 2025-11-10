#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test script to verify PI0OpenPI policy integration with FastVLA vs the original implementation, only meant to be run locally!"""

import os

import pytest
import torch
import gc

# Skip if openpi or transformers is not available
pytest.importorskip("openpi")
pytest.importorskip("transformers")


# NOTE: Assumes PYTHONPATH is set to include OpenPI src as per instructions.
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from fastvla.policies.pi05.model_loader import instantiate_fastvla_pi05
from fastvla.policies.pi05.preprocessing import create_original_observation_with_openpi_preprocessing
from fastvla.policies.pi05.config import DUMMY_ACTION_DIM, DUMMY_STATE_DIM, DUMMY_ACTION_HORIZON, PI05BaseOriginalConfig

# TODO: ADDING DEFAULT IMAGES_FEATURES TO CONFIG

DEVICE = "cpu"  # Use CPU to avoid memory issues for testing

DUMMY_DATASET_STATS = {
    "observation.state": {
        "mean": torch.zeros(DUMMY_STATE_DIM),
        "std": torch.ones(DUMMY_STATE_DIM),
        "q01": torch.zeros(DUMMY_STATE_DIM),
        "q99": torch.ones(DUMMY_STATE_DIM),
    },
    "action": {
        "mean": torch.zeros(DUMMY_ACTION_DIM),
        "std": torch.ones(DUMMY_ACTION_DIM),
        "q01": torch.zeros(DUMMY_ACTION_DIM),
        "q99": torch.ones(DUMMY_ACTION_DIM),
    },
    "images": {
        "base_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "left_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
        "right_wrist_0_rgb": {
            "mean": torch.zeros(3, 224, 224),
            "std": torch.ones(3, 224, 224),
            "q01": torch.zeros(3, 224, 224),
            "q99": torch.ones(3, 224, 224),
        },
    },
}


def instantiate_original_pi05(from_pretrained: bool = False, model_path: str | None = None, device="cuda"):
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


def create_dummy_data(device: str = "cuda"):
    batch_size = 2  # Reduce batch size for testing

    # Use the exact same prompt for both implementations
    prompt = "Pick up the red block and place it in the bin"

    batch = {
        "observation.state": torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, dtype=torch.float32, device=device),
        # Create images in [0, 1] range as expected by FastVLA (will be converted to [-1, 1] internally)
        "observation.images.base_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        "observation.images.left_wrist_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        "observation.images.right_wrist_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        # Add the task prompt for FastVLA - provide as list with single element to trigger expansion
        "task": [prompt for _ in range(batch_size)],
    }
    return batch


def extract_fastvla_processed_inputs(fastvla_pi0, batch):
    """Extract the exact same processed inputs that FastVLA uses internally."""
    # Get the tokenized language from FastVLA's internal method
    lang_tokens, lang_masks = fastvla_pi0._tokenize_language(batch)

    # Get the preprocessed images from FastVLA's internal method
    images, img_masks = fastvla_pi0._preprocess_images(batch, train=False)

    # Create dummy token_ar_mask and token_loss_mask for original implementation
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    return images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask


class PI05Observation:
    """Observation class that matches the original OpenPI format."""

    def __init__(
        self,
        state,
        images,
        image_masks,
        tokenized_prompt,
        tokenized_prompt_mask,
        token_ar_mask,
        token_loss_mask,
    ):
        self.state = state
        self.images = images
        self.image_masks = image_masks
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


def create_original_observation_from_fastvla(fastvla_pi0, batch):
    """Create observation object compatible with original OpenPI using the exact same inputs as FastVLA."""
    _batch_size = batch["observation.state"].shape[0]
    _device = batch["observation.state"].device

    # Extract the exact same processed inputs that FastVLA uses
    images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask = extract_fastvla_processed_inputs(
        fastvla_pi0, batch
    )

    # Convert images list to dict with original OpenPI keys
    image_dict = {
        "base_0_rgb": images[0],
        "left_wrist_0_rgb": images[1],
        "right_wrist_0_rgb": images[2],
    }

    # Convert image masks list to dict with original OpenPI keys
    image_masks_dict = {
        "base_0_rgb": img_masks[0],
        "left_wrist_0_rgb": img_masks[1],
        "right_wrist_0_rgb": img_masks[2],
    }

    return PI05Observation(
        state=batch["observation.state"],
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )


def set_torch_random_seed(seed: int):
    """Set the random seed for torch for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_pi05_original_vs_fastvla():
    device = "cuda"
    """Test PI05 original implementation vs Fastvla implementation."""
    set_torch_random_seed(42)  # Set the same seed
    original_pi0 = instantiate_original_pi05(
        from_pretrained=False, device=device
    )  # Load pretrained OpenPI model from HuggingFace Hub
    set_torch_random_seed(42)  # Set the same seed
    print("Creating dummy data...")
    batch = create_dummy_data(device)

    # Test each model with its own preprocessing (more realistic end-to-end test)
    print("Creating observation for OpenPI using OpenPI's own preprocessing...")
    pi0_obs_openpi = create_original_observation_with_openpi_preprocessing(batch)

    print(f"Task prompt: '{batch['task'][0]}'")
    print(f"OpenPI tokenized prompt shape: {pi0_obs_openpi.tokenized_prompt.shape}")
    print(f"OpenPI image shapes: {[img.shape for img in pi0_obs_openpi.images.values()]}")
    print(f"OpenPI state shape: {pi0_obs_openpi.state.shape}")

    print("Testing OpenPI with own preprocessing...")
    original_pi0.eval()
    torch.manual_seed(42)  # Set seed for reproducibility
    batch_size = batch["observation.state"].shape[0]
    noise_shape = (batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM)
    fixed_noise = torch.randn(noise_shape, dtype=torch.float32, device=device)

    with torch.no_grad():
        openpi_actions = original_pi0.sample_actions(
            device=device, observation=pi0_obs_openpi, noise=fixed_noise, num_steps=10
        ).clone()
        openpi_actions_unit = openpi_actions[:, 0, :]
    print(f"OpenPI (own preprocessing) Actions shape: {openpi_actions.shape}")
    print(f"OpenPI (own preprocessing) Actions unit shape: {openpi_actions_unit.shape}")
    print(f"OpenPI (own preprocessing) Actions mean: {openpi_actions.mean().item():.6f}")
    print(f"OpenPI (own preprocessing) Actions std: {openpi_actions.std().item():.6f}")

    # release memory before running models
    del original_pi0
    gc.collect()
    torch.cuda.empty_cache()

    print("Testing FastVLA with own preprocessing...")
    set_torch_random_seed(42)  # Set the same seed
    fastvla_pi05 = instantiate_fastvla_pi05(from_pretrained=False, device=device)  # Load pretrained FastVLA model
    fastvla_pi05.eval()
    with torch.no_grad():
        fastvla_actions_own = fastvla_pi05.sample_actions(
            device=device, observation=pi0_obs_openpi, noise=fixed_noise, num_steps=10
        ).clone()  # batch_size, n_action_steps, action_dim
        fastvla_actions_unit = fastvla_actions_own[:, 0, :]
    print(f"FastVLA (own preprocessing) Actions shape: {fastvla_actions_own.shape}")
    print(f"FastVLA (own preprocessing) Actions unit shape: {fastvla_actions_unit.shape}")
    print(f"FastVLA (own preprocessing) Actions mean: {fastvla_actions_own.mean().item():.6f}")
    print(f"FastVLA (own preprocessing) Actions std: {fastvla_actions_own.std().item():.6f}")

    print("\nComparing end-to-end implementations:")
    print(f"Actions close (atol=1e-4): {torch.allclose(fastvla_actions_own, openpi_actions, atol=1e-4)}")
    print(f"Actions close (atol=1e-2): {torch.allclose(fastvla_actions_own, openpi_actions, atol=1e-2)}")
    print(f"Max absolute difference: {torch.abs(fastvla_actions_own - openpi_actions).max().item():.6f}")
    if fastvla_pi05.config.dtype == "bfloat16":
        tolerance = 1e-2
    elif fastvla_pi05.config.dtype == "float32":
        tolerance = 1e-4 if device == "cpu" else 1e-3
    assert torch.allclose(fastvla_actions_own, openpi_actions, atol=tolerance)
