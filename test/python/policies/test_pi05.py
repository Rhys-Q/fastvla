from fastvla.policies.pi05.modeling_pi05 import PI05Policy


def test_pi05_policy_load():
    policy = PI05Policy.from_pretrained("lerobot/pi05_base", strict=True, load_weights=True)
    print("Successfully loaded PI05 policy model with pretrained weights.")


def test_pi05_policy_config_only():
    policy = PI05Policy.from_pretrained("lerobot/pi05_base", strict=True, load_weights=False)
    # Sanity check: parameters should exist but not require any specific loaded key
    n_params = sum(p.numel() for p in policy.parameters())
    assert n_params > 0
    print("Successfully initialized PI05 policy model without pretrained weights (random init).")


if __name__ == "__main__":
    test_pi05_policy_load()
    test_pi05_policy_config_only()
