from fastvla.policies.pi05.model_loader import instantiate_fastvla_pi05
import pytest
import gc


def test_pi05_policy_load():
    policy = instantiate_fastvla_pi05(from_pretrained=True, device="cuda")
    print("PI05 policy instantiated with pretrained weights.")
    del policy
    gc.collect()

    policy = instantiate_fastvla_pi05(from_pretrained=False, device="cuda")
    print("PI05 policy instantiated with random weights.")
    del policy
    gc.collect()


if __name__ == "__main__":
    pytest.main([__file__])
