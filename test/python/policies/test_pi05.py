from fastvla.policies.pi05.modeling_pi05 import PI05Policy


def test_pi05_policy_load():
    policy = PI05Policy.from_pretrained("lerobot/pi05_base", strict=True)
    print("Successfully loaded PI05 policy model.")


if __name__ == "__main__":
    test_pi05_policy_load()
