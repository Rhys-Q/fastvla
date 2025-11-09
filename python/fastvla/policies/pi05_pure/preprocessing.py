from collections.abc import Sequence
import logging
from transformers import AutoTokenizer  # noqa: E402
import torch
from copy import deepcopy
import numpy as np
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)
DUMMY_MAX_TOKEN_LEN = 200
DUMMY_ACTION_DIM = 32
DUMMY_STATE_DIM = 32
DUMMY_ACTION_HORIZON = 50
DUMMY_MAX_TOKEN_LEN = 200


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        # Convert to channels-first for torch operations
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images, size=(resized_height, resized_width), mode=mode, align_corners=False if mode == "bilinear" else None
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]
        if batch_size == 1 and images.shape[0] == 1:
            padded_images = padded_images.squeeze(0)  # Remove batch dimension if it was added

    return padded_images


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(
                    0, 2, 3, 1
                )  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(
                        0, 2, 3, 1
                    )  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            image = image * brightness_factor

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


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


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


def create_original_observation_with_openpi_preprocessing(batch):
    """Create observation object for OpenPI using OpenPI's own preprocessing with pi05 state tokenizer."""
    batch_size = batch["observation.state"].shape[0]
    device = batch["observation.state"].device

    # Create tokenizer for OpenPI (same as LeRobot uses)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

    # Get task description (pi05 processor handles all text formatting)
    tasks = batch.get("task", ["Pick up the object"] * batch_size)
    if isinstance(tasks, str):
        tasks = [tasks] * batch_size
    elif len(tasks) == 1:
        tasks = tasks * batch_size

    # Use pi05 state and input tokenizer logic (same as Pi05PrepareStateTokenizerProcessorStep)
    state = batch["observation.state"]
    state = deepcopy(state)

    # Prepare state (pad to max_state_dim)

    state = pad_vector(state, DUMMY_STATE_DIM)

    # Normalize state to [-1, 1] range if needed (assuming it's already normalized from normalize_inputs)
    # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
    state_np = state.cpu().numpy()
    discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

    # Create pi05-formatted prompts that include state information
    full_prompts = []
    for i, task in enumerate(tasks):
        cleaned_text = task.strip().replace("_", " ").replace("\n", " ")
        state_str = " ".join(map(str, discretized_states[i]))
        full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
        full_prompts.append(full_prompt)

    # Tokenize with max_length padding to match OpenPI's expected format
    tokenized = tokenizer(
        full_prompts,
        padding="max_length",
        padding_side="right",
        truncation=True,
        max_length=DUMMY_MAX_TOKEN_LEN,
        return_tensors="pt",
    )

    lang_tokens = tokenized["input_ids"].to(device)
    lang_masks = tokenized["attention_mask"].to(device, dtype=torch.bool)

    # Create dummy token_ar_mask and token_loss_mask for OpenPI
    token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    # Convert LeRobot images format to OpenPI format (convert [0,1] to [-1,1] range)
    image_dict = {
        "base_0_rgb": batch["observation.images.base_0_rgb"] * 2.0 - 1.0,
        "left_wrist_0_rgb": batch["observation.images.left_wrist_0_rgb"] * 2.0 - 1.0,
        "right_wrist_0_rgb": batch["observation.images.right_wrist_0_rgb"] * 2.0 - 1.0,
    }

    # Create image masks (all ones for real images)
    image_masks_dict = {}
    for key in image_dict:
        image_masks_dict[key] = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Create raw observation object (before preprocessing)
    raw_observation = PI05Observation(
        state=batch["observation.state"],
        images=image_dict,
        image_masks=image_masks_dict,
        tokenized_prompt=lang_tokens,
        tokenized_prompt_mask=lang_masks,
        token_ar_mask=token_ar_mask,
        token_loss_mask=token_loss_mask,
    )

    # Now use OpenPI's preprocessing
    processed_obs = preprocess_observation_pytorch(raw_observation, train=False)

    return processed_obs
