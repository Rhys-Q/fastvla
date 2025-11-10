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
import time


# NOTE: Assumes PYTHONPATH is set to include OpenPI src as per instructions.
from fastvla.policies.pi05.model_loader import instantiate_fastvla_pi05
from fastvla.policies.pi05.preprocessing import create_original_observation_with_openpi_preprocessing
from fastvla.policies.pi05.config import DUMMY_ACTION_DIM, DUMMY_STATE_DIM, DUMMY_ACTION_HORIZON, PI05BaseOriginalConfig

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


def create_dummy_data(device: str = "cuda"):
    batch_size = 1  # Reduce batch size for testing

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
    print("Creating dummy data...")
    batch = create_dummy_data(device)

    # Test each model with its own preprocessing (more realistic end-to-end test)
    print("Creating observation for OpenPI using OpenPI's own preprocessing...")
    pi0_obs_openpi = create_original_observation_with_openpi_preprocessing(batch)

    torch.manual_seed(42)  # Set seed for reproducibility
    batch_size = batch["observation.state"].shape[0]
    noise_shape = (batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM)
    fixed_noise = torch.randn(noise_shape, dtype=torch.float32, device=device)

    set_torch_random_seed(42)  # Set the same seed
    fastvla_pi05 = instantiate_fastvla_pi05(from_pretrained=False, device=device)  # Load pretrained FastVLA model
    fastvla_pi05.eval()
    # profile memory usage and latency
    with torch.no_grad():
        # warm up
        for _ in range(5):
            fastvla_pi05.sample_actions(device=device, observation=pi0_obs_openpi, noise=fixed_noise, num_steps=10)
        # profile memory usage and latency
        latencies = []
        for _ in range(20):
            start_time = time.time()
            fastvla_pi05.sample_actions(
                device=device, observation=pi0_obs_openpi, noise=fixed_noise, num_steps=10
            ).clone()
            end_time = time.time()
            latencies.append(end_time - start_time)
        avg_latency = sum(latencies) / len(latencies)
        print(f"FastVLA PI05 average latency over 20 runs: {avg_latency}")
