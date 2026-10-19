"""Verify the machine can run the voice agent on the GPU.

Run:  uv run check-env
Exit code 0 = all required checks passed, 1 = at least one FAIL.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from voice_agent.config import Config, load_config

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
MIN_PYTHON = (3, 12)
MIN_DRIVER = 570  # CUDA 12.8 wheels need driver >= 570


@dataclass
class Check:
    name: str
    status: str
    detail: str


# ---------------------------------------------------------------- system

def check_python() -> Check:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return Check("Python", PASS if ok else FAIL, f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")


def is_wsl() -> bool:
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def check_platform() -> Check:
    desc = f"{platform.system()} {platform.release()} ({platform.machine()})"
    if platform.system() == "Windows":
        return Check("Platform", PASS, f"native {desc}")
    if is_wsl():
        return Check("Platform", WARN, f"WSL2 {desc}: target is native Windows")
    return Check("Platform", WARN, f"{desc}: target is native Windows")


def check_config(cfg_path: Path | None = None) -> tuple[Check, Config | None]:
    try:
        cfg = load_config(cfg_path) if cfg_path else load_config()
        cfg.paths.models_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        return Check("Config", PASS, f"device={cfg.device}, models_dir={cfg.paths.models_dir}"), cfg
    except Exception as e:  # noqa: BLE001
        return Check("Config", FAIL, f"{type(e).__name__}: {e}"), None


def parse_driver_version(text: str) -> int | None:
    m = re.search(r"(\d+)\.\d+", text)
    return int(m.group(1)) if m else None


def check_driver() -> Check:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return Check("NVIDIA driver", FAIL, "nvidia-smi not found: install the NVIDIA driver")
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return Check("NVIDIA driver", FAIL, f"nvidia-smi failed: {e}")
    major = parse_driver_version(out)
    if major is None:
        return Check("NVIDIA driver", WARN, f"could not parse: {out}")
    status = PASS if major >= MIN_DRIVER else FAIL
    return Check("NVIDIA driver", status, f"{out} (need >= {MIN_DRIVER})")


# ---------------------------------------------------------------- torch / GPU

def check_torch_gpu(torch: ModuleType, gpu_index: int = 0) -> list[Check]:
    """All GPU checks. Takes the torch module so tests can pass a fake."""
    checks = [Check("PyTorch", PASS, f"{torch.__version__}, CUDA runtime {torch.version.cuda}")]

    if torch.version.cuda is None:
        checks.append(Check("CUDA build", FAIL, "CPU-only torch installed: run `uv sync` so the cu128 index is used"))
        return checks
    if not torch.cuda.is_available():
        checks.append(Check("CUDA available", FAIL, "torch.cuda.is_available() is False: check driver / GPU"))
        return checks

    dev = torch.device(f"cuda:{gpu_index}")
    props = torch.cuda.get_device_properties(dev)
    vram_gb = props.total_memory / 1024**3
    checks.append(Check("GPU", PASS, f"{props.name}, {vram_gb:.1f} GB VRAM"))

    major, minor = props.major, props.minor
    arch = f"sm_{major}{minor}"
    arch_list = torch.cuda.get_arch_list()
    arch_ok = arch in arch_list or f"compute_{major}{minor}" in arch_list
    checks.append(Check(
        "Kernel support", PASS if arch_ok else FAIL,
        f"GPU is {arch}; torch built for {', '.join(arch_list)}",
    ))
    if not arch_ok:
        return checks

    checks.append(check_matmul(torch, dev))
    bf16 = torch.cuda.is_bf16_supported()
    checks.append(Check("bfloat16", PASS if bf16 else WARN, "supported" if bf16 else "not supported"))
    return checks


def check_matmul(torch: ModuleType, dev, n: int = 4096, iters: int = 20) -> Check:
    """fp16 matmul on GPU: correctness vs fp32 reference + rough throughput."""
    try:
        torch.manual_seed(0)
        a = torch.randn(n, n, device=dev)
        b = torch.randn(n, n, device=dev)
        ref = a @ b
        a16, b16 = a.half(), b.half()
        out = a16 @ b16  # warm-up
        err = ((out.float() - ref).abs().max() / ref.abs().max()).item()

        sync = (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            a16 @ b16
        sync()
        dt = (time.perf_counter() - t0) / iters
        tflops = 2 * n**3 / dt / 1e12
        ok = err < 1e-2
        return Check("fp16 matmul", PASS if ok else FAIL, f"rel err {err:.1e}, {tflops:.0f} TFLOPS")
    except Exception as e:  # noqa: BLE001
        return Check("fp16 matmul", FAIL, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- runner

def run_all(cfg_path: Path | None = None) -> list[Check]:
    checks = [check_python(), check_platform()]
    cfg_check, cfg = check_config(cfg_path)
    checks.append(cfg_check)

    if cfg is not None and cfg.device == "cpu":
        checks.append(Check("GPU", WARN, "device=cpu in config: GPU checks skipped"))
        return checks

    checks.append(check_driver())
    try:
        import torch
    except ImportError as e:
        checks.append(Check("PyTorch", FAIL, f"not importable: {e}"))
        return checks
    checks.extend(check_torch_gpu(torch, cfg.gpu_index if cfg else 0))
    if torch.cuda.is_available():
        checks.extend(check_libraries())
    return checks


def check_libraries() -> list[Check]:
    """GPU libraries: CTranslate2 (Whisper) and torchaudio/transformers (XTTS)."""
    out = []
    try:
        from voice_agent import cuda_dlls

        cuda_dlls.prepare()
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        types = sorted(ctranslate2.get_supported_compute_types("cuda")) if n else []
        ok = n > 0 and "float16" in types
        out.append(Check("CTranslate2", PASS if ok else FAIL,
                         f"{ctranslate2.__version__}, {n} CUDA device(s), compute types {types}"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("CTranslate2", FAIL, f"{type(e).__name__}: {e}"))
    try:
        import torchaudio
        import transformers

        out.append(Check("Audio libs", PASS, f"torchaudio {torchaudio.__version__}, transformers {transformers.__version__}"))
    except Exception as e:  # noqa: BLE001
        out.append(Check("Audio libs", FAIL, f"{type(e).__name__}: {e}"))
    out.append(check_onnxruntime())
    return out


def check_onnxruntime(versions: dict | None = None, providers: list[str] | None = None) -> Check:
    """Kokoro (English TTS) runs on ONNX Runtime. It needs the GPU build to use CUDA.

    Pipecat also installs the CPU build ("onnxruntime"). Both write the same folder,
    so the CPU one can replace the GPU files. Then Kokoro silently runs on the CPU.
    """
    import importlib.metadata as md

    if versions is None:
        versions = {}
        for dist in ("onnxruntime", "onnxruntime-gpu"):
            try:
                versions[dist] = md.version(dist)
            except md.PackageNotFoundError:
                pass
    if providers is None:
        try:
            import onnxruntime as rt

            providers = rt.get_available_providers()
        except Exception as e:  # noqa: BLE001
            return Check("ONNX Runtime", FAIL, f"not importable: {e}")
    has_cuda = "CUDAExecutionProvider" in providers
    info = f"{versions}, providers {providers}"
    if has_cuda:
        return Check("ONNX Runtime", PASS, info)
    fix = ("fix: `uv pip uninstall onnxruntime onnxruntime-gpu` then "
           f"`uv pip install onnxruntime-gpu=={versions.get('onnxruntime-gpu', '1.30.0')}`")
    return Check("ONNX Runtime", WARN, f"no CUDA provider, Kokoro will use the CPU. {info}. {fix}")


def print_report(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.status}] {c.name:<{width}}  {c.detail}")
    fails = sum(c.status == FAIL for c in checks)
    warns = sum(c.status == WARN for c in checks)
    print("-" * 60)
    print("RESULT:", "FAIL" if fails else "OK", f"({fails} fail, {warns} warn)")


def main() -> int:
    checks = run_all()
    print_report(checks)
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
