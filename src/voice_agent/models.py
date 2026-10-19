"""Offline model store.

  uv run download-models            # download every model (needs internet, once)
  uv run download-models whisper-large-v3-turbo
  uv run verify-models              # offline check: every model present on disk

Runtime code calls `enable_offline()` so nothing ever reaches the network.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

from voice_agent.config import PROJECT_ROOT, load_config

MODELS_YAML = PROJECT_ROOT / "config" / "models.yaml"
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def enable_offline() -> None:
    """Block Hugging Face network access for this process."""
    os.environ.update(OFFLINE_ENV)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "hf" | "bundled" | "url"
    used_by: str = ""
    repo: str = ""
    required_files: list[str] = field(default_factory=list)
    package: str = ""
    file: str = ""
    allow_patterns: list[str] | None = None  # download only these files
    gated: bool = False      # needs HF terms accepted + `hf auth login`
    optional: bool = False   # missing = WARN, not FAIL
    urls: dict = field(default_factory=dict)  # kind "url": file name -> download link


def load_registry(path: Path = MODELS_YAML) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["models"]
    specs = {}
    for name, d in raw.items():
        spec = ModelSpec(name=name, **d)
        if spec.kind not in ("hf", "bundled", "url"):
            raise ValueError(f"{name}: unknown kind {spec.kind!r}")
        if spec.kind == "hf" and not spec.repo:
            raise ValueError(f"{name}: 'hf' model needs 'repo'")
        if spec.kind == "url" and not spec.urls:
            raise ValueError(f"{name}: 'url' model needs 'urls'")
        if spec.kind == "bundled" and not (spec.package and spec.file):
            raise ValueError(f"{name}: 'bundled' model needs 'package' and 'file'")
        specs[name] = spec
    return specs


def model_path(spec: ModelSpec, models_dir: Path | None = None) -> Path:
    """Where the model lives on disk."""
    if spec.kind == "bundled":
        return Path(str(resources.files(spec.package).joinpath(spec.file)))
    models_dir = models_dir or load_config().paths.models_dir
    return models_dir / spec.name


def _size_mb(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def verify(spec: ModelSpec, models_dir: Path) -> tuple[bool, str]:
    try:
        p = model_path(spec, models_dir)
    except ModuleNotFoundError as e:
        return False, f"package missing: {e}"
    if spec.kind == "bundled":
        return (p.is_file(), f"{p.name} {_size_mb(p):.1f} MB (bundled)" if p.is_file() else f"missing {p}")
    if not p.is_dir():
        return False, f"not downloaded ({p})"
    missing = [f for f in (spec.required_files or list(spec.urls)) if not (p / f).is_file()]
    if missing:
        return False, f"incomplete, missing {missing}"
    return True, f"{_size_mb(p):.0f} MB at {p}"


def download(spec: ModelSpec, models_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    dest = models_dir / spec.name
    if spec.kind == "url":
        return _download_urls(spec, dest)
    # local_dir = plain files, no symlinks (safe on Windows); resumes if interrupted.
    snapshot_download(repo_id=spec.repo, local_dir=str(dest), allow_patterns=spec.allow_patterns)
    return dest


def _download_urls(spec: ModelSpec, dest: Path) -> Path:
    # This function downloads plain files (e.g. GitHub release files) into models/<name>/.
    import httpx

    dest.mkdir(parents=True, exist_ok=True)
    for name, url in spec.urls.items():
        target = dest / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        tmp = target.with_suffix(target.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for block in r.iter_bytes(1 << 20):
                    f.write(block)
        tmp.replace(target)
    return dest


def _select(specs: dict[str, ModelSpec], names: list[str]) -> list[ModelSpec]:
    unknown = [n for n in names if n not in specs]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Known: {list(specs)}")
    return [specs[n] for n in names] if names else list(specs.values())


def download_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download models for offline use")
    ap.add_argument("names", nargs="*", help="model names (default: all non-gated)")
    args = ap.parse_args(argv)
    specs = load_registry()
    models_dir = load_config().paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    for spec in _select(specs, args.names):
        if spec.kind == "bundled":
            print(f"[skip] {spec.name}: bundled with pip package")
            continue
        if spec.gated and spec.name not in args.names:
            print(f"[skip] {spec.name}: gated, run `download-models {spec.name}` after `hf auth login`")
            continue
        print(f"[get ] {spec.name} <- {spec.repo or 'github release'}")
        try:
            download(spec, models_dir)
        except Exception as e:  # noqa: BLE001
            hint = " (gated: accept terms on huggingface.co, then `uv run hf auth login`)" if spec.gated else ""
            print(f"[FAIL] {spec.name}: {type(e).__name__}: {e}{hint}")
            continue
        ok, detail = verify(spec, models_dir)
        print(f"[{'ok' if ok else 'FAIL'}  ] {spec.name}: {detail}")
    return 0


def verify_main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check all models are available offline")
    ap.add_argument("names", nargs="*")
    args = ap.parse_args(argv)
    enable_offline()
    from loguru import logger

    logger.remove()  # hide pipecat's import banner
    specs = load_registry()
    models_dir = load_config().paths.models_dir

    failed = 0
    for spec in _select(specs, args.names):
        ok, detail = verify(spec, models_dir)
        status = "PASS" if ok else ("WARN" if spec.optional else "FAIL")
        failed += status == "FAIL"
        print(f"[{status}] {spec.name:<26} {detail:<60} ({spec.used_by})")
    print("-" * 60)
    print("RESULT:", "FAIL" if failed else "OK", f"({failed} missing)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(verify_main())
