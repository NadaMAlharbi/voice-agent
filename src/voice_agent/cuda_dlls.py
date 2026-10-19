"""Make CUDA DLLs visible to non-torch libraries on Windows.

CTranslate2 (faster-whisper) needs the cuBLAS and cuDNN DLLs. The CUDA build of torch
already includes them, so no separate CUDA install is needed. Does nothing on Linux/macOS.
"""

from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path


@cache
def prepare() -> None:
    if sys.platform != "win32":
        return
    import torch  # noqa: F401  (loads cublas/cudnn DLLs into the process)

    lib = Path(torch.__file__).parent / "lib"
    if lib.is_dir():
        os.add_dll_directory(str(lib))
        os.environ["PATH"] = f"{lib}{os.pathsep}{os.environ.get('PATH', '')}"
