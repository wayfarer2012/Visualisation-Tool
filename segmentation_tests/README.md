# SAM 3.1-Only Wall and Ceiling Test

This script runs SAM 3.1 outside the lightweight PySide6 environment. The main
app launches it as a subprocess with the text prompts `wall` and `ceiling`.
It does not use Grounding DINO or SAM 2.1.

## Hardware Note

The laptop's GTX 1660 Ti supports CUDA, but SAM 3.1 is a large model and may
exceed its available VRAM. The script catches CUDA out-of-memory errors and
suggests reducing the input image resolution.

## Official SAM 3.1 Requirements

- Windows 11 with an NVIDIA CUDA-compatible GPU
- Python 3.12 or newer
- PyTorch 2.7 or newer with CUDA support
- CUDA 12.6 or newer
- Latest official Meta SAM 3 repository code
- Approved access to the gated `facebook/sam3.1` Hugging Face checkpoint

The SAM 3.1 checkpoint repository contains checkpoints only. It is not
integrated into Hugging Face Transformers.

## Install Into `.venv-sam31`

The error `ModuleNotFoundError: No module named 'sam3'` means the official SAM 3
repository has not been installed into `.venv-sam31`.

The official Meta installation method is to clone `facebookresearch/sam3` and
install the cloned repository as an editable Python package. Run these commands
from the project root:

```powershell
cd "C:\Users\Wayfarer2012\Documents\Visualisation Tool"

New-Item -ItemType Directory -Path external -Force
git clone https://github.com/facebookresearch/sam3.git external\sam3

.\.venv-sam31\Scripts\python.exe -m pip install --upgrade pip
.\.venv-sam31\Scripts\python.exe -m pip install -e external\sam3
.\.venv-sam31\Scripts\python.exe -m pip install -r segmentation_tests\requirements.txt
```

SAM 3.1 requires the latest SAM 3 repository code. If `external\sam3` already
exists, update and reinstall it instead:

```powershell
git -C external\sam3 pull
.\.venv-sam31\Scripts\python.exe -m pip install -e external\sam3
```

Verify that the SAM 3 package and the exact imports used by the test script are
available inside `.venv-sam31`:

```powershell
.\.venv-sam31\Scripts\python.exe -c "import sam3; from sam3.model_builder import build_sam3_image_model; from sam3.model.sam3_image_processor import Sam3Processor; print('SAM 3 imports are working')"
```

The active test script uses these official imports:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

Installing the repository into `.venv-sam31` does not modify the main PySide6
application or its existing environment.

## Request Model Access

1. Sign in and request access at:
   https://huggingface.co/facebook/sam3.1
2. Create a Hugging Face access token.
3. Authenticate:

```powershell
.\.venv-sam31\Scripts\hf.exe auth login
```

The script downloads `sam3.1_multiplex.pt` from `facebook/sam3.1`. No manual
checkpoint placement is required. Installing the SAM 3 Python package and
receiving/authenticating for checkpoint access are separate requirements.

## Check The GPU

```powershell
.\.venv-sam31\Scripts\python.exe segmentation_tests\check_gpu.py
```

The check prints Python, PyTorch, CUDA, GPU name, and total GPU memory. If
PyTorch is missing, it prints a friendly message instead of crashing.

## Place The Input Image

The required input path is:

```text
segmentation_tests/input_room.jpg
```

An `input_image.jpeg` file can be copied or renamed with:

```powershell
Copy-Item segmentation_tests\input_image.jpeg segmentation_tests\input_room.jpg
```

## Run

Run the original isolated test with its default input/output paths:

```powershell
.\.venv-sam31\Scripts\python.exe segmentation_tests\test_sam31_wall_ceiling.py
```

The main app uses this form after creating a visualisation session:

```powershell
.\.venv-sam31\Scripts\python.exe segmentation_tests\test_sam31_wall_ceiling.py --input "<visualisation_folder>\original.png" --output "<visualisation_folder>"
```

## Expected Outputs

```text
<visualisation_folder>/
  original.png
  combined_preview.png
  segments.json
  masks/
    segment_001.png
    segment_002.png
```

`segments.json` contains app-compatible editable pre-cuts: segment IDs,
`wall`/`ceiling` types, `shape_type: "mask"`, relative mask paths, and the
saved `applied_colour` value. Reopening a visualisation uses these files and
does not rerun SAM.

## Interior And Exterior Prompting

The script prompts SAM 3.1 for interior walls and ceilings plus exterior
paintable architectural surfaces such as facade walls, porch walls, gables,
parapets, pillars, columns, beams, trim, fascia, soffits, and roof undersides.
Related wall prompts normalize to the app type `wall`; distinct architectural
parts keep types such as `pillar`, `column`, `beam`, `trim`, and `soffit`.

It also prompts for common non-paintable regions such as windows, doors,
glass, sky, vegetation, roads, cars, floors, ground, roof tiles, and shingles.
A target mask is removed only when an exclusion mask covers most of it. This
conservative rule avoids deleting a large wall merely because windows or doors
are present inside it. Strongly overlapping target masks are deduplicated using
intersection-over-union before final mask files and `segments.json` are saved.

## Common Failures

- **No checkpoint/model access:** Accept the gated model terms and run
  `hf auth login`.
- **CUDA unavailable:** Install a CUDA-enabled PyTorch build and verify the
  NVIDIA driver using `check_gpu.py`.
- **CUDA out of memory:** Reduce `input_room.jpg` resolution. The GTX 1660 Ti
  has limited VRAM for SAM 3.1.
- **Wrong Python version:** Create the environment with Python 3.12 or newer.
- **PyTorch/CUDA mismatch:** Reinstall the official CUDA PyTorch wheel matching
  the setup command above.
- **SAM 3.1 checkpoint loading issue:** Pull the latest official SAM 3 code and
  reinstall it. SAM 3.1 requires newer code than earlier SAM 3 checkpoints.

SAM 2.1 may be considered as a future fallback only. It is not implemented or
used by this experiment.
