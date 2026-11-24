"""Project-local paths for model downloads and runtime caches."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".cache"

_CACHE_PATHS = {
    "HF_HOME": CACHE_DIR / "huggingface",
    "HF_HUB_CACHE": CACHE_DIR / "huggingface" / "hub",
    "HF_ASSETS_CACHE": CACHE_DIR / "huggingface" / "assets",
    "TORCH_HOME": CACHE_DIR / "torch",
    "XDG_CACHE_HOME": CACHE_DIR,
    "MPLCONFIGDIR": CACHE_DIR / "matplotlib",
}

for _name, _path in _CACHE_PATHS.items():
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_name] = str(_path)
