"""Convert isolated SAM 3.1 output into a mock app visualisation session.

SAM output segments are model-level detections containing confidence and bbox
data. App segments are visualisation-level editable pre-cuts that belong to one
saved room visualisation and can later store an applied colour.

This bridge does not modify or connect to app.py. Later, app.py can be extended
to load these mask-based segments instead of generating fake polygons.
"""

import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Conversion paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEGMENTATION_TESTS_DIR = PROJECT_ROOT / "segmentation_tests"
SAM_OUTPUT_DIR = SEGMENTATION_TESTS_DIR / "output"
SAM_SEGMENTS_PATH = SAM_OUTPUT_DIR / "segments.json"

CONVERTED_JSON_PATH = SAM_OUTPUT_DIR / "converted_app_segments.json"
MOCK_VISUALISATION_DIR = SAM_OUTPUT_DIR / "app_visualisation_mock"
MOCK_MASKS_DIR = MOCK_VISUALISATION_DIR / "masks"
MOCK_ORIGINAL_PATH = MOCK_VISUALISATION_DIR / "original.png"
MOCK_SEGMENTS_PATH = MOCK_VISUALISATION_DIR / "segments.json"

VISUALISATION_ID = "test_sam31_conversion"
SUPPORTED_TYPES = {
    "wall",
    "ceiling",
    "pillar",
    "column",
    "parapet",
    "gable",
    "soffit",
    "fascia",
    "beam",
    "trim",
}


def load_sam_output() -> dict:
    """Load and validate the successful SAM 3.1 segmentation result."""
    if not SAM_SEGMENTS_PATH.exists():
        raise FileNotFoundError(f"SAM output JSON not found: {SAM_SEGMENTS_PATH}")

    with SAM_SEGMENTS_PATH.open("r", encoding="utf-8") as json_file:
        document = json.load(json_file)

    if not isinstance(document.get("segments"), list):
        raise ValueError("SAM output JSON must contain a segments list.")
    return document


def resolve_project_path(path_text: str) -> Path:
    """Resolve paths written relative to the shared project root."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def copy_original_as_png(source_image_path: Path) -> None:
    """Copy the original test room image into the mock session as original.png."""
    if not source_image_path.exists():
        raise FileNotFoundError(f"Original input image not found: {source_image_path}")

    # Saving through an image library guarantees that original.png really uses
    # PNG bytes instead of being a JPEG file with a renamed extension.
    try:
        from PIL import Image

        with Image.open(source_image_path) as source_image:
            source_image.convert("RGB").save(MOCK_ORIGINAL_PATH, format="PNG")
        return
    except ImportError:
        pass

    try:
        from PySide6.QtGui import QImage
    except ImportError as error:
        raise RuntimeError(
            "Converting the original image to PNG requires Pillow or PySide6."
        ) from error

    source_image = QImage(str(source_image_path))
    if source_image.isNull() or not source_image.save(str(MOCK_ORIGINAL_PATH), "PNG"):
        raise OSError(f"Could not convert image to PNG: {source_image_path}")


def convert_segments(sam_segments: list[dict]) -> list[dict]:
    """Copy SAM masks and create visualisation-level editable segment records."""
    converted_segments = []

    for index, sam_segment in enumerate(sam_segments, start=1):
        segment_id = sam_segment.get("id") or f"segment_{index:03d}"
        segment_type = sam_segment.get("type")
        source_mask_text = sam_segment.get("mask_path")

        if segment_type not in SUPPORTED_TYPES:
            print(f"Skipping {segment_id}: unsupported type {segment_type!r}")
            continue
        if not source_mask_text:
            print(f"Skipping {segment_id}: mask_path is missing")
            continue

        source_mask_path = Path(source_mask_text)
        if not source_mask_path.is_absolute():
            # New app-compatible output uses masks/ relative to its output
            # folder. The fallback retains support for older output/masks paths.
            output_relative_path = SAM_OUTPUT_DIR / source_mask_path
            legacy_path = SEGMENTATION_TESTS_DIR / source_mask_path
            source_mask_path = (
                output_relative_path
                if output_relative_path.exists()
                else legacy_path
            )
        if not source_mask_path.exists():
            print(f"Skipping {segment_id}: mask file not found at {source_mask_path}")
            continue

        destination_mask_name = f"{segment_id}.png"
        destination_mask_path = MOCK_MASKS_DIR / destination_mask_name
        shutil.copy2(source_mask_path, destination_mask_path)

        # The mask path is relative to the future visualisation session folder.
        converted_segments.append(
            {
                "id": segment_id,
                "type": segment_type,
                "shape_type": "mask",
                "mask_path": f"masks/{destination_mask_name}",
                "applied_colour": None,
            }
        )

    return converted_segments


def write_json(document: dict, output_path: Path) -> None:
    """Write a readable JSON document."""
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(document, json_file, indent=2)


def main() -> None:
    """Build converted JSON and a complete mock app visualisation folder."""
    sam_output = load_sam_output()

    MOCK_MASKS_DIR.mkdir(parents=True, exist_ok=True)
    # New app-compatible SAM output sits beside original.png and does not need
    # an image_path field. Older isolated output remains supported.
    if sam_output.get("image_path"):
        source_image_path = resolve_project_path(sam_output["image_path"])
    elif (SAM_OUTPUT_DIR / "original.png").exists():
        source_image_path = SAM_OUTPUT_DIR / "original.png"
    else:
        source_image_path = SEGMENTATION_TESTS_DIR / "input_room.jpg"
    copy_original_as_png(source_image_path)

    converted_document = {
        "visualisation_id": VISUALISATION_ID,
        "segments": convert_segments(sam_output["segments"]),
    }

    # This standalone file makes the conversion result easy to inspect.
    write_json(converted_document, CONVERTED_JSON_PATH)

    # This copy sits beside original.png and masks/, matching a future app session.
    write_json(converted_document, MOCK_SEGMENTS_PATH)

    print(f"Converted {len(converted_document['segments'])} segment(s).")
    print(f"Converted JSON: {CONVERTED_JSON_PATH}")
    print(f"Mock visualisation folder: {MOCK_VISUALISATION_DIR}")


if __name__ == "__main__":
    main()
