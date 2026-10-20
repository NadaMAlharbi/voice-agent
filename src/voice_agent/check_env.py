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
        if cfg is not None:
            checks.append(check_kokoro(cfg))
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


def _varint(n: int) -> bytes:
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _field(num: int, value) -> bytes:
    """One protobuf field: int -> varint, bytes/str -> length-delimited."""
    if isinstance(value, int):
        return _varint(num << 3) + _varint(value)
    data = value.encode() if isinstance(value, str) else value
    return _varint((num << 3) | 2) + _varint(len(data)) + data


def tiny_matmul_onnx(n: int = 64) -> bytes:
    """A one-node ONNX model, Y = A @ B (float [n, n]), built without the onnx package.

    MatMul on the CUDA provider calls cuBLAS, so running it proves the CUDA,
    cuBLAS/cuBLASLt and cuDNN DLLs that onnxruntime-gpu was built against really load.
    """
    def value_info(name: str) -> bytes:
        dims = b"".join(_field(1, _field(1, n)) for _ in range(2))          # Dimension.dim_value
        tensor = _field(1, 1) + _field(2, dims)                              # elem_type FLOAT, shape
        return _field(1, name) + _field(2, _field(1, tensor))                # name, TypeProto.tensor_type
    node = _field(1, "A") + _field(1, "B") + _field(2, "Y") + _field(4, "MatMul")
    graph = (_field(1, node) + _field(2, "probe") + _field(11, value_info("A"))
             + _field(11, value_info("B")) + _field(12, value_info("Y")))
    opset = _field(1, "") + _field(2, 13)
    return _field(1, 8) + _field(8, opset) + _field(7, graph)               # ir_version 8, opset 13


def probe_ort_cuda() -> tuple[list[str], float]:
    """Run a real MatMul on the CUDA provider. Returns (session providers, max abs error)."""
    import numpy as np
    import onnxruntime as rt

    from voice_agent import cuda_dlls

    cuda_dlls.prepare()
    so = rt.SessionOptions()
    so.log_severity_level = 3            # the CUDA load error is reported by us, not printed twice
    sess = rt.InferenceSession(tiny_matmul_onnx(), so,
                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    a, b = (rng.standard_normal((64, 64), dtype=np.float32) for _ in range(2))
    (y,) = sess.run(None, {"A": a, "B": b})
    return sess.get_providers(), float(np.abs(y - a @ b).max())


def check_onnxruntime(versions: dict | None = None, providers: list[str] | None = None,
                      probe=probe_ort_cuda) -> Check:
    """Kokoro (English TTS) runs on ONNX Runtime. It needs the GPU build to use CUDA.

    Two ways it silently ends up on the CPU:
      - pipecat/faster-whisper/kokoro-onnx want the CPU build ("onnxruntime"). Both
        write the same folder, so the CPU one can replace the GPU files.
        pyproject.toml drops it with [tool.uv] override-dependencies.
      - onnxruntime-gpu is built for another CUDA major than torch (PyPI 1.27+ is
        CUDA 13, torch cu128 is CUDA 12): "cublasLt64_13.dll is missing".
    get_available_providers() lists CUDA in both cases, so we build a real session.
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
    info = f"{versions}, providers {providers}"
    fix = "fix: `uv sync --reinstall-package onnxruntime-gpu` (pyproject pins a CUDA 12 build)"
    if "onnxruntime" in versions and "onnxruntime-gpu" in versions:
        return Check("ONNX Runtime", WARN, f"CPU and GPU builds both installed, they overwrite each other. {info}. "
                     "fix: `uv sync` (removes the CPU build), then "
                     "`uv sync --reinstall-package onnxruntime-gpu`")
    if "CUDAExecutionProvider" not in providers:
        return Check("ONNX Runtime", WARN, f"no CUDA provider, Kokoro will use the CPU. {info}. {fix}")
    if probe is None:
        return Check("ONNX Runtime", PASS, info)
    from voice_agent.tts.kokoro_engine import cuda_mismatch_hint

    gpu_version = versions.get("onnxruntime-gpu", "")
    try:
        got, err = probe()
    except Exception as e:  # noqa: BLE001
        return Check("ONNX Runtime", WARN, f"CUDA test session failed ({type(e).__name__}: {e}). {info}. "
                     f"{cuda_mismatch_hint(gpu_version)} {fix}")
    if not got or got[0] != "CUDAExecutionProvider":
        return Check("ONNX Runtime", WARN, f"CUDA provider did not load (session uses {got}), Kokoro will use "
                     f"the CPU. {info}. {cuda_mismatch_hint(gpu_version)} {fix}")
    if err > 1e-2:
        return Check("ONNX Runtime", FAIL, f"CUDA MatMul result is wrong (max error {err:.3g}). {info}")
    return Check("ONNX Runtime", PASS, f"{info}; CUDA MatMul test OK")


def check_kokoro(cfg: Config) -> Check:
    """Load Kokoro the way the server does and run one real synthesis on the GPU."""
    try:
        from voice_agent.models import load_registry, model_path

        spec = load_registry()[cfg.tts.kokoro.get("model", "kokoro")]
        folder = model_path(spec, cfg.paths.models_dir)
        if not (folder / "kokoro-v1.0.onnx").is_file():
            return Check("Kokoro GPU", WARN, f"model not downloaded ({folder}); run `uv run download-models`")
        from kokoro_onnx import Kokoro

        from voice_agent.tts.kokoro_engine import LANG_CODE, make_session

        sess = make_session(folder / "kokoro-v1.0.onnx", cfg.device)
        model = Kokoro.from_session(sess, str(folder / "voices-v1.0.bin"))
        voice = sorted(model.get_voices())[0]
        model.create("Hello.", voice=voice, lang=LANG_CODE)          # warm-up (CUDA kernels)
        t0 = time.perf_counter()
        audio, sr = model.create("This sentence checks the GPU.", voice=voice, lang=LANG_CODE)
        ms = (time.perf_counter() - t0) * 1000
        got = sess.get_providers()
        detail = f"providers {got}, {len(audio) / sr:.1f}s audio in {ms:.0f} ms"
        ok = bool(got) and got[0] == "CUDAExecutionProvider"
        return Check("Kokoro GPU", PASS if ok else WARN, detail if ok else f"runs on the CPU: {detail}")
    except Exception as e:  # noqa: BLE001
        return Check("Kokoro GPU", WARN, f"{type(e).__name__}: {e}")


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
