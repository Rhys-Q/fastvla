DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200


class PI05BaseOriginalConfig:
    action_dim: int = DUMMY_ACTION_DIM
    action_horizon: int = DUMMY_ACTION_HORIZON
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    precision: str = "float32"
    pi05: bool = True
    dtype: str = "float32"
