"""Print the local Python, PyTorch, CUDA, and GPU configuration."""

import sys


def main() -> None:
    """Show GPU information without crashing when PyTorch is missing."""
    print(f"Python version: {sys.version}")

    try:
        import torch
    except ImportError:
        print("PyTorch: missing")
        print("Install PyTorch with CUDA support before running SAM 3.1.")
        return

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version reported by PyTorch: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("GPU name: unavailable")
        print("Total GPU memory: unavailable")
        print("SAM 3.1 requires a CUDA-compatible GPU for this experiment.")
        return

    properties = torch.cuda.get_device_properties(0)
    total_memory_gb = properties.total_memory / (1024**3)
    print(f"GPU name: {properties.name}")
    print(f"Total GPU memory: {total_memory_gb:.2f} GB")


if __name__ == "__main__":
    main()
