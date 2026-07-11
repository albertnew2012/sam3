"""
Shared setup helpers for the SAM 3 study scripts.

Every study script imports from here so the boilerplate (TF32, autocast,
locating the repo assets, and a place to drop outputs) lives in one spot.
"""

import os

# Reduce CUDA fragmentation. Must be set before the first CUDA allocation; the
# allocator reads it lazily, so setting it at import time (before any model runs)
# is sufficient. Harmless on CPU.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

# Repo root = one level above the `sam3` package directory.
import sam3

SAM3_ROOT = os.path.abspath(os.path.join(os.path.dirname(sam3.__file__), ".."))
ASSETS = os.path.join(SAM3_ROOT, "assets")
# The BPE vocab ships inside the installed package; the builders resolve it
# automatically when bpe_path=None, so scripts never need to pass it.

# Where every script writes its visualizations. Kept out of git via .gitignore.
OUT_DIR = os.path.join(SAM3_ROOT, "study", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def setup_runtime():
    """Enable the fast paths SAM 3 expects and enter a bf16 autocast context.

    Returns the torch.device the model should live on. Call this once at the
    top of a script. The autocast context is entered process-wide (never
    exited) which mirrors what the official notebooks do.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # TensorFloat-32 gives a big speedup on Ampere+ (the RTX 3090 here).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.autocast("cuda", dtype=dtype).__enter__()
        print(f"[runtime] device=cuda dtype={dtype} gpu={torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[runtime] device=cpu (SAM 3 is trained on CUDA; expect this to be slow)")
    return device


def out(name: str) -> str:
    """Absolute path inside the study outputs dir for a result file."""
    return os.path.join(OUT_DIR, name)
