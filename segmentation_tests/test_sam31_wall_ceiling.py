"""SAM 3.1-only paintable architectural surface segmentation.

This file does not import or modify the main PySide6 visualisation application.
It supports interior and exterior paintable surfaces with text prompts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image


# ---------------------------------------------------------------------------
# Paths and prompts
# ---------------------------------------------------------------------------

INPUT_IMAGE_PATH = Path("segmentation_tests/input_room.jpg")
OUTPUT_DIR = Path("segmentation_tests/output")

# Choose "interior", "exterior", or "auto". Auto preserves the working
# interior prompts while also finding broad exterior paintable zones.
SEGMENTATION_MODE = "auto"

# Fewer broad prompts produce cleaner paintable zones than a detailed
# architectural taxonomy. Each group maps to one stable app segment type.
EXTERIOR_PROMPT_GROUPS = {
    "wall surfaces": [
        ("paintable exterior wall surface", "wall"),
        ("house facade wall", "wall"),
        ("large flat exterior wall area", "wall"),
    ],
    "columns and pillars": [
        ("paintable column", "pillar"),
        ("paintable pillar", "pillar"),
    ],
    "trims, soffits, and fascia": [
        ("paintable trim", "trim"),
        ("paintable fascia", "trim"),
        ("paintable soffit", "trim"),
        ("paintable roof underside", "trim"),
    ],
}
INTERIOR_PROMPT_GROUPS = {
    "interior fallback": [
        ("interior wall", "wall"),
        ("ceiling", "ceiling"),
    ]
}

# Exclusion prompts are used only as a conservative post-processing check.
# A target is rejected only when an excluded mask covers most of that target.
EXCLUSION_PROMPTS = [
    "window",
    "door",
    "glass",
    "sky",
    "tree",
    "plant",
    "grass",
    "driveway",
    "car",
    "floor",
    "ground",
    "roof tile",
    "shingle",
]

# This gated checkpoint requires approved Hugging Face access and `hf auth login`.
SAM31_REPOSITORY = "facebook/sam3.1"
SAM31_CHECKPOINT_FILE = "sam3.1_multiplex.pt"

CONFIDENCE_THRESHOLD = 0.25
DUPLICATE_IOU_THRESHOLD = 0.72
DUPLICATE_CONTAINMENT_THRESHOLD = 0.85
EXCLUDED_COVERAGE_THRESHOLD = 0.65
MINIMUM_MASK_AREA_RATIO = 0.001
MINIMUM_MASK_FILL_RATIO = 0.12
MINIMUM_BBOX_DIMENSION = 4

# GTX 1660 Ti is a pre-Ampere GPU and does not handle SAM 3.1's faster BF16
# path reliably. Keep CUDA enabled, but run model weights and inference in FP32.
USE_FLOAT32 = True

PREVIEW_COLOURS = {
    "wall": np.array([255, 70, 70], dtype=np.float32),
    "ceiling": np.array([70, 145, 255], dtype=np.float32),
    "pillar": np.array([255, 220, 70], dtype=np.float32),
    "trim": np.array([255, 110, 190], dtype=np.float32),
}

ORIGINAL_TORCH_AUTOCAST = torch.autocast


def force_float32_tensors(value):
    """Recursively convert floating-point tensors and nested values to FP32."""
    if isinstance(value, torch.Tensor):
        return value.float() if value.is_floating_point() else value
    if isinstance(value, dict):
        return {key: force_float32_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [force_float32_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(force_float32_tensors(item) for item in value)
    return value


def force_model_parameters_and_buffers_float32(model) -> None:
    """Force all floating model parameters and buffers to CUDA float32."""
    model.float()

    # Be explicit because SAM 3.1 checkpoints and nested modules can contain
    # floating buffers that are not obvious from the top-level model.
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()

    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer.is_floating_point():
                module._buffers[name] = buffer.float()


def install_float32_input_hooks(model) -> list:
    """Force every floating activation entering a model module back to FP32.

    SAM 3.1 contains internal BF16-oriented paths intended for newer Ampere
    GPUs. The GTX 1660 Ti is pre-Ampere, so this test keeps CUDA but converts
    any BF16 activation before it reaches a Float32 layer.
    """

    def float32_pre_hook(module, args, kwargs):
        return force_float32_tensors(args), force_float32_tensors(kwargs)

    return [
        module.register_forward_pre_hook(float32_pre_hook, with_kwargs=True)
        for module in model.modules()
    ]


def disable_cuda_autocast() -> None:
    """Disable any CUDA autocast state left active by imported SAM internals."""
    try:
        torch.set_autocast_enabled("cuda", False)
    except TypeError:
        # Compatibility path for PyTorch versions using the older signature.
        torch.set_autocast_enabled(False)


def install_bf16_autocast_guard() -> None:
    """Neutralize hard-coded CUDA BF16 autocast requests inside SAM 3.1.

    This is a runtime-only test-script guard. It avoids modifying external/sam3
    while preventing its Ampere-focused BF16 contexts on the GTX 1660 Ti.
    """

    def guarded_autocast(device_type, dtype=None, enabled=True, cache_enabled=None):
        if device_type == "cuda" and dtype == torch.bfloat16:
            return ORIGINAL_TORCH_AUTOCAST(
                device_type=device_type,
                dtype=dtype,
                enabled=False,
                cache_enabled=cache_enabled,
            )
        return ORIGINAL_TORCH_AUTOCAST(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=cache_enabled,
        )

    torch.autocast = guarded_autocast
    disable_cuda_autocast()


def print_model_dtype_debug(model) -> None:
    """Print representative parameter and buffer dtypes after FP32 conversion."""
    first_parameter_name, first_parameter = next(model.named_parameters())
    print(
        "First model parameter: "
        f"{first_parameter_name} | dtype={first_parameter.dtype} | "
        f"device={first_parameter.device}"
    )

    printed_buffers = 0
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            print(f"Model buffer: {name} | dtype={buffer.dtype} | device={buffer.device}")
            printed_buffers += 1
            if printed_buffers >= 3:
                break

    non_float32_parameters = [
        name
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.float32
    ]
    non_float32_buffers = [
        name
        for name, buffer in model.named_buffers()
        if buffer.is_floating_point() and buffer.dtype != torch.float32
    ]
    print(f"Non-FP32 floating parameters: {len(non_float32_parameters)}")
    print(f"Non-FP32 floating buffers: {len(non_float32_buffers)}")
    try:
        print(f"CUDA autocast active: {torch.is_autocast_enabled('cuda')}")
    except TypeError:
        print(f"CUDA autocast active: {torch.is_autocast_enabled()}")


def install_backbone_input_debug_hook(model):
    """Print the image tensor dtype received by the visual backbone."""

    def debug_backbone_input(module, args, kwargs):
        tensors = []
        force_float32_tensors(args)

        def collect(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    collect(item)

        collect(args)
        collect(kwargs)
        for index, tensor in enumerate(tensors[:3], start=1):
            print(
                f"Backbone input tensor {index}: dtype={tensor.dtype}, "
                f"device={tensor.device}, shape={tuple(tensor.shape)}"
            )

    return model.backbone.register_forward_pre_hook(debug_backbone_input, with_kwargs=True)


def tensor_to_numpy(value) -> np.ndarray:
    """Move a model tensor to CPU and convert it to a NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_mask(mask) -> np.ndarray:
    """Convert one returned mask into a two-dimensional boolean array."""
    mask_array = np.squeeze(tensor_to_numpy(mask))
    if mask_array.ndim != 2:
        raise ValueError(f"Unexpected SAM 3.1 mask shape: {mask_array.shape}")
    return mask_array > 0


def mask_bounding_box(mask: np.ndarray) -> list[int] | None:
    """Calculate a JSON-safe [x, y, width, height] box from a mask."""
    y_positions, x_positions = np.where(mask)
    if len(x_positions) == 0:
        return None

    x_min = int(x_positions.min())
    x_max = int(x_positions.max())
    y_min = int(y_positions.min())
    y_max = int(y_positions.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def normalize_score(score) -> float | None:
    """Convert an optional model confidence score to a Python float."""
    if score is None:
        return None
    return round(float(tensor_to_numpy(score).reshape(-1)[0]), 4)


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Save one boolean mask as a black-and-white PNG."""
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def mask_iou(first_mask: np.ndarray, second_mask: np.ndarray) -> float:
    """Return intersection-over-union for duplicate-mask filtering."""
    intersection = np.logical_and(first_mask, second_mask).sum()
    union = np.logical_or(first_mask, second_mask).sum()
    return float(intersection / union) if union else 0.0


def mask_covered_by(mask: np.ndarray, covering_mask: np.ndarray) -> float:
    """Return how much of one candidate is covered by an excluded region."""
    mask_area = mask.sum()
    if not mask_area:
        return 0.0
    return float(np.logical_and(mask, covering_mask).sum() / mask_area)


def mask_fill_ratio(mask: np.ndarray) -> float:
    """Measure how continuously a mask fills its bounding box."""
    bounding_box = mask_bounding_box(mask)
    if bounding_box is None:
        return 0.0
    _, _, width, height = bounding_box
    return float(mask.sum() / (width * height))


def mask_quality_score(detection: dict) -> float:
    """Prefer larger, continuous, confident masks before deduplication."""
    mask = detection["mask"]
    confidence = detection.get("confidence")
    confidence_factor = confidence if confidence is not None else 1.0
    return float(mask.sum() * (0.5 + 0.5 * mask_fill_ratio(mask)) * confidence_factor)


def selected_prompt_groups() -> dict[str, list[tuple[str, str]]]:
    """Return broad prompt groups for the configured segmentation mode."""
    if SEGMENTATION_MODE == "interior":
        return INTERIOR_PROMPT_GROUPS
    if SEGMENTATION_MODE == "exterior":
        return EXTERIOR_PROMPT_GROUPS
    if SEGMENTATION_MODE == "auto":
        return {**EXTERIOR_PROMPT_GROUPS, **INTERIOR_PROMPT_GROUPS}
    raise ValueError(
        'SEGMENTATION_MODE must be "interior", "exterior", or "auto".'
    )


def filter_paintable_detections(
    detections: list[dict],
    exclusion_detections: list[dict],
    image_size: tuple[int, int],
) -> list[dict]:
    """Keep clean paintable zones and remove fragments, noise, and duplicates."""
    image_area = image_size[0] * image_size[1]
    minimum_area = image_area * MINIMUM_MASK_AREA_RATIO
    accepted = []
    removed_counts = {"small": 0, "noisy": 0, "excluded": 0, "duplicate": 0}

    # Larger and cleaner candidates are considered first. If two candidates
    # overlap heavily, the stronger one remains and the fragment is discarded.
    for detection in sorted(detections, key=mask_quality_score, reverse=True):
        mask = detection["mask"]
        if mask.sum() < minimum_area:
            print(f'Filtered tiny mask from prompt: {detection["prompt"]}')
            removed_counts["small"] += 1
            continue

        bounding_box = detection.get("bbox") or mask_bounding_box(mask)
        _, _, width, height = bounding_box
        fill_ratio = mask_fill_ratio(mask)
        if (
            min(width, height) < MINIMUM_BBOX_DIMENSION
            or fill_ratio < MINIMUM_MASK_FILL_RATIO
        ):
            print(
                "Filtered thin/noisy mask: "
                f'{detection["prompt"]} (fill={fill_ratio:.2f})'
            )
            removed_counts["noisy"] += 1
            continue

        excluded_match = next(
            (
                excluded
                for excluded in exclusion_detections
                if mask_covered_by(mask, excluded["mask"])
                >= EXCLUDED_COVERAGE_THRESHOLD
            ),
            None,
        )
        if excluded_match is not None:
            print(
                "Filtered excluded region: "
                f'{detection["prompt"]} overlaps {excluded_match["prompt"]}'
            )
            removed_counts["excluded"] += 1
            continue

        duplicate_match = next(
            (
                existing
                for existing in accepted
                if (
                    mask_iou(mask, existing["mask"]) >= DUPLICATE_IOU_THRESHOLD
                    or (
                        detection["type"] == existing["type"]
                        and mask_covered_by(mask, existing["mask"])
                        >= DUPLICATE_CONTAINMENT_THRESHOLD
                    )
                )
            ),
            None,
        )
        if duplicate_match is not None:
            print(
                "Filtered duplicate mask: "
                f'{detection["prompt"]} overlaps {duplicate_match["prompt"]}'
            )
            removed_counts["duplicate"] += 1
            continue

        accepted.append(detection)

    print(
        "Filtering summary: "
        f'{removed_counts["small"]} small, '
        f'{removed_counts["noisy"]} thin/noisy, '
        f'{removed_counts["excluded"]} excluded, '
        f'{removed_counts["duplicate"]} duplicate mask(s) removed'
    )
    return accepted


def create_combined_preview(
    image: Image.Image, segments: list[dict], output_path: Path
) -> None:
    """Blend detected masks over the original image for easy visual inspection."""
    preview = np.asarray(image, dtype=np.float32).copy()

    for segment in segments:
        mask = segment["_mask"]
        colour = PREVIEW_COLOURS.get(
            segment["type"], np.array([255, 255, 255], dtype=np.float32)
        )
        preview[mask] = preview[mask] * 0.45 + colour * 0.55

    Image.fromarray(np.clip(preview, 0, 255).astype(np.uint8)).save(output_path)


def build_sam31_model():
    """Download the approved SAM 3.1 checkpoint and build the image model."""
    if USE_FLOAT32:
        # Install this before importing SAM modules, because some SAM functions
        # create BF16 autocast contexts at import or construction time.
        install_bf16_autocast_guard()

    # Import the official SAM package only after CUDA has been checked.
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    checkpoint_path = hf_hub_download(
        repo_id=SAM31_REPOSITORY,
        filename=SAM31_CHECKPOINT_FILE,
    )

    # Supplying the explicit checkpoint path prevents accidental SAM 3.0 loading.
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
    )

    if USE_FLOAT32:
        # The builder has no public dtype argument. Convert all checkpoint
        # weights after loading so BF16 weights cannot mix with FP32 inputs.
        force_model_parameters_and_buffers_float32(model)

    return model, Sam3Processor


def segment_text_prompt(
    processor, image_state, prompt: str, segment_type: str
) -> list[dict]:
    """Run one SAM 3.1 text prompt and normalize its returned masks."""
    if USE_FLOAT32:
        # Explicitly disable CUDA autocast. In particular, never enter the
        # torch.bfloat16 autocast path on the GTX 1660 Ti.
        disable_cuda_autocast()
        image_state = force_float32_tensors(image_state)
        with torch.autocast(device_type="cuda", enabled=False):
            output = processor.set_text_prompt(state=image_state, prompt=prompt)
    else:
        output = processor.set_text_prompt(state=image_state, prompt=prompt)
    masks = output.get("masks", [])
    scores = output.get("scores", [])
    detections = []
    print(f'Raw masks for "{prompt}": {len(masks)}')

    for index, raw_mask in enumerate(masks):
        confidence = normalize_score(scores[index]) if index < len(scores) else None
        if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
            continue

        mask = normalize_mask(raw_mask)
        detections.append(
            {
                "type": segment_type,
                "prompt": prompt,
                "mask": mask,
                "bbox": mask_bounding_box(mask),
                "confidence": confidence,
            }
        )

    print(f'Confidence-kept masks for "{prompt}": {len(detections)}')
    return detections


def run_segmentation(input_image_path: Path, output_dir: Path) -> int:
    """Detect interior and exterior paintable surfaces and save app pre-cuts."""
    if not input_image_path.exists():
        raise FileNotFoundError(
            f"Input image missing: {input_image_path}\n"
            "Copy segmentation_tests/input_image.jpeg to "
            "segmentation_tests/input_room.jpg."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This SAM 3.1 experiment requires a "
            "CUDA-compatible GPU and CUDA-enabled PyTorch."
        )

    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(input_image_path).convert("RGB")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    if USE_FLOAT32:
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        disable_cuda_autocast()
        print("Precision mode: CUDA float32 (BF16 autocast disabled)")

    print("Loading the gated facebook/sam3.1 checkpoint...")
    model, processor_class = build_sam31_model()
    float32_hooks = []
    backbone_debug_hook = None
    if USE_FLOAT32:
        # Keep these hook handles alive for the complete inference run.
        float32_hooks = install_float32_input_hooks(model)
        backbone_debug_hook = install_backbone_input_debug_hook(model)
        print_model_dtype_debug(model)

    processor = processor_class(model)

    # Encode the room once and reuse that state for both text prompts.
    if USE_FLOAT32:
        # Sam3Processor already prepares image tensors as float32. This disabled
        # autocast context and recursive conversion keep the full state in FP32.
        with torch.autocast(device_type="cuda", enabled=False):
            image_state = processor.set_image(image)
        image_state = force_float32_tensors(image_state)
        print("Image state floating tensors converted to float32.")
    else:
        image_state = processor.set_image(image)
    print(f"Segmentation mode: {SEGMENTATION_MODE}")
    target_detections = []
    for group_name, prompts in selected_prompt_groups().items():
        print(f"Running prompt group: {group_name}")
        for prompt, segment_type in prompts:
            print(f"Running prompt: {prompt}")
            target_detections.extend(
                segment_text_prompt(processor, image_state, prompt, segment_type)
            )

    exclusion_detections = []
    for prompt in EXCLUSION_PROMPTS:
        print(f"Running exclusion prompt: {prompt}")
        exclusion_detections.extend(
            segment_text_prompt(processor, image_state, prompt, prompt)
        )

    # SAM output is model-level detection data. Filtering turns it into useful,
    # visualisation-level editable pre-cuts before masks are written for the app.
    detections = filter_paintable_detections(
        target_detections, exclusion_detections, image.size
    )
    kept_counts = {}
    for detection in detections:
        kept_counts[detection["type"]] = kept_counts.get(detection["type"], 0) + 1
    for segment_type in sorted(kept_counts):
        print(f"Final kept masks for type {segment_type}: {kept_counts[segment_type]}")

    segments = []
    for index, detection in enumerate(detections, start=1):
        segment_id = f"segment_{index:03d}"
        mask_relative_path = Path("masks") / f"{segment_id}.png"
        mask_path = output_dir / mask_relative_path
        save_mask(detection["mask"], mask_path)

        # SAM detections become visualisation-level editable pre-cuts. The app
        # later updates only applied_colour while preserving these mask paths.
        segments.append(
            {
                "id": segment_id,
                "type": detection["type"],
                "shape_type": "mask",
                "mask_path": mask_relative_path.as_posix(),
                "applied_colour": None,
                "_mask": detection["mask"],
            }
        )

    create_combined_preview(image, segments, output_dir / "combined_preview.png")

    json_segments = [
        {key: value for key, value in segment.items() if key != "_mask"}
        for segment in segments
    ]
    with (output_dir / "segments.json").open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "visualisation_id": output_dir.name,
                "segments": json_segments,
            },
            json_file,
            indent=2,
        )

    print(f"Detected {len(segments)} final paintable surface segment(s).")
    print(f"Outputs saved to: {output_dir.resolve()}")

    # References are intentionally kept until inference is complete.
    del float32_hooks, backbone_debug_hook
    return len(segments)


def parse_arguments() -> argparse.Namespace:
    """Read optional paths used by the main app's external subprocess call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_IMAGE_PATH,
        help="Room image to segment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Visualisation folder that will receive masks/ and segments.json.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the experiment with friendly CUDA out-of-memory guidance."""
    arguments = parse_arguments()
    try:
        run_segmentation(arguments.input, arguments.output)
        return 0
    except torch.cuda.OutOfMemoryError:
        print(
            "\nCUDA ran out of memory while running SAM 3.1.\n"
            "SAM ran out of GPU memory. Try using a smaller image.",
            file=sys.stderr,
        )
        torch.cuda.empty_cache()
        return 2
    except RuntimeError as error:
        if "out of memory" in str(error).lower():
            print(
                "\nCUDA ran out of memory while running SAM 3.1.\n"
                "SAM ran out of GPU memory. Try using a smaller image.",
                file=sys.stderr,
            )
            torch.cuda.empty_cache()
            return 2
        else:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
