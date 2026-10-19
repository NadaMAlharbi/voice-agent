"""CPU, RAM, disk, battery and GPU information of this computer."""

from __future__ import annotations

import platform
import shutil
import subprocess
from functools import partial
from pathlib import Path

from voice_agent.tools.base import Tool, ToolResult


def gpu_info() -> dict:
    # This function reads GPU name, load, memory and temperature with nvidia-smi (empty if no NVIDIA GPU).
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        out = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
        name, util, used, total, temp = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"gpu": name, "gpu_percent": int(util), "gpu_memory_used_gb": round(int(used) / 1024, 1),
                "gpu_memory_total_gb": round(int(total) / 1024, 1), "gpu_temp_c": int(temp)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def system_status(tb=None) -> ToolResult:
    # This function collects CPU, RAM, disk, battery and GPU numbers.
    import psutil

    d = {"cpu_percent": psutil.cpu_percent(interval=0.3), "memory_percent": psutil.virtual_memory().percent,
         "memory_total_gb": round(psutil.virtual_memory().total / 2**30, 1), "os": platform.platform(terse=True)}
    try:
        d["disk_percent"] = psutil.disk_usage("C:\\" if Path("C:\\").exists() else "/").percent
    except OSError:
        pass
    b = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if b is not None:
        d["battery_percent"] = round(b.percent)
        d["charging"] = bool(b.power_plugged)
    d.update(gpu_info())
    return ToolResult(True, d)


def tools(tb) -> list[Tool]:
    return [Tool("system_status", "CPU, memory, disk, battery and GPU status of this computer.", {},
                 partial(system_status, tb))]
