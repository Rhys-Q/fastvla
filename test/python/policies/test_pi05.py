from typing import Any
from fastvla.policies.pi05.modeling_pi05 import PI05Policy
from fastvla.policies.pi05 import PI05Config, PI05Policy  # noqa: E402
from fastvla.policies.pi05.processor_pi05 import make_pi05_pre_post_processors  # noqa: E402
from fastvla.processor import PolicyAction, PolicyProcessorPipeline  # noqa: E402
from fastvla.policies import pi05_pure
import torch
import pytest


def test_pi05_policy_load():
    policy = PI05Policy.from_pretrained("lerobot/pi05_base", strict=True, load_weights=True)
    print("Successfully loaded PI05 policy model with pretrained weights.")


def test_pi05_policy_config_only():
    policy = PI05Policy.from_pretrained("lerobot/pi05_base", strict=True, load_weights=False)
    # Sanity check: parameters should exist but not require any specific loaded key
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params > 0
    print("Successfully initialized PI05 policy model without pretrained weights (random init).")


# TODO: ADDING DEFAULT IMAGES_FEATURES TO CONFIG
DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200
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


def instantiate_lerobot_pi05(
    from_pretrained: bool = False,
    load_weights: bool = True,
) -> tuple[
    PI05Policy,
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    if from_pretrained:
        # Load the policy first
        policy = PI05Policy.from_pretrained(
            pretrained_name_or_path="lerobot/pi05_base", strict=True, load_weights=load_weights
        )
    else:
        config = PI05Config(max_action_dim=DUMMY_ACTION_DIM, max_state_dim=DUMMY_STATE_DIM, dtype="float32")
        policy = PI05Policy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    preprocessor, postprocessor = make_pi05_pre_post_processors(config=policy.config, dataset_stats=DUMMY_DATASET_STATS)
    return (policy, preprocessor, postprocessor)


def create_dummy_data():
    batch_size = 2  # Reduce batch size for testing
    device = DEVICE

    # Use the exact same prompt for both implementations
    prompt = "Pick up the red block and place it in the bin"

    batch = {
        "observation.state": torch.randn(batch_size, DUMMY_STATE_DIM, dtype=torch.float32, device=device),
        "action": torch.randn(batch_size, DUMMY_ACTION_HORIZON, DUMMY_ACTION_DIM, dtype=torch.float32, device=device),
        # Create images in [0, 1] range as expected by LeRobot (will be converted to [-1, 1] internally)
        "observation.images.base_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        "observation.images.left_wrist_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        "observation.images.right_wrist_0_rgb": torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        # Add the task prompt for LeRobot - provide as list with single element to trigger expansion
        "task": [prompt for _ in range(batch_size)],
    }
    return batch


def test_pi05_policy_run():
    policy, preprocessor, postprocessor = instantiate_lerobot_pi05(from_pretrained=True, load_weights=False)
    policy.eval()
    batch = create_dummy_data()
    batch_lerobot_processed = preprocessor(batch)
    with torch.no_grad():
        lerobot_actions_own = policy.predict_action_chunk(
            batch_lerobot_processed
        )  # batch_size, n_action_steps, action_dim
        lerobot_actions_unit = lerobot_actions_own[:, 0, :]
    print("PI05 policy forward pass successful. Action shape:", lerobot_actions_unit.shape)
    print("Sample actions:", lerobot_actions_unit)


def test_pi05_pure_load():
    policy = pi05_pure.PI05Policy.from_pretrained("lerobot/pi05_base", strict=True, load_weights=False)
    # Sanity check: parameters should exist but not require any specific loaded key
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params > 0
    print("Successfully initialized PI05 policy model without pretrained weights (random init).")


if __name__ == "__main__":
    pytest.main([__file__])
