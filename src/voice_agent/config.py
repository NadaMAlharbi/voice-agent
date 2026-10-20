"""Load config/config.yaml into typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    models_dir: Path
    logs_dir: Path


@dataclass(frozen=True)
class Server:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(frozen=True)
class Audio:
    input_sample_rate: int = 16000
    output_sample_rate: int = 24000
    frame_ms: int = 20


@dataclass(frozen=True)
class Vad:
    confidence: float = 0.7
    start_secs: float = 0.2
    stop_secs: float = 0.2
    min_volume: float = 0.72


@dataclass(frozen=True)
class Turn:
    stop_secs: float = 1.0
    pre_speech_ms: float = 500
    max_duration_secs: float = 8


@dataclass(frozen=True)
class Stt:
    engine: str = "whisper"
    language: str = "auto"           # auto | ar | en
    whisper: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Llm:
    host: str = "http://127.0.0.1:11434"
    model: str = "qwen3.5:9b"
    think: bool = False
    keep_alive: int | str = -1
    temperature: float = 0.6
    top_p: float = 0.8               # Qwen3.5 suggestion for non-thinking mode
    top_k: int = 20                  # fewer unlikely tokens: less drift into other languages
    presence_penalty: float = 0.0    # > 0 discourages repeating words (Qwen suggests up to 1.5)
    max_tokens: int = 200
    system_prompt: str = ""
    num_ctx: int = 8192              # tool schemas + history need > Ollama's 4k default; same value for
                                     # every request (a different num_ctx makes Ollama reload the model)
    cache_friendly: bool = True      # per-turn notes in the user turn: system prompt + tools stay KV-cached


@dataclass(frozen=True)
class Tts:
    language: str = "en"             # active voice: en | ar
    gender: str = "female"           # female | male
    engine: str = "models"           # models (Kokoro + XTTS) | tone (test)
    voices: dict = field(default_factory=dict)
    kokoro: dict = field(default_factory=dict)
    xtts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunker:
    first_min_chars: int = 12        # first chunk may end at a comma once this long
    min_chars: int = 40              # later chunks: merge short sentences up to this
    soft_min_chars: int = 60         # later chunks may end at a comma once this long
    max_chars: int = 220             # hard split at a space beyond this
    max_chars_ar: int = 160          # Arabic: XTTS may cut audio above 166 characters
    first_min_chars_ar: int = 30     # Arabic: XTTS restarts its voice on every chunk, so the first
    min_chars_ar: int = 80           # chunk is not a 1-2 word scrap and short sentences are merged
    first_min_words: int = 3         # a sentence end closes the first chunk only after N words
    first_max_words: int = 6         # first chunk cut after N whole words if no punctuation yet (0 = off)
    second_max_words: int = 16       # second chunk cap, so it is ready before chunk 1 finishes
    first_max_words_ar: int = 5      # Arabic words are ~1.4x longer spoken: same audio length, fewer words
    second_max_words_ar: int = 11    # (0 = use the English caps)


@dataclass(frozen=True)
class BargeIn:
    min_speech_s: float = 0.6        # while the assistant speaks: speech needed to interrupt
    echo_threshold: float = 0.6      # mic/assistant spectral match treated as echo
    max_echo_lag_s: float = 0.6      # speaker -> mic delay searched
    min_volume: float = 0.74         # while the assistant speaks: quieter (far) voices do not interrupt


@dataclass(frozen=True)
class Tools:
    enabled: bool = True
    desktop: str = "auto"            # auto | windows | x11 | dry-run (records actions, touches nothing)
    browser: str = "chrome"          # chrome | edge (falls back to the default browser)
    screenshots_dir: Path = Path.home() / "Pictures" / "HAZEM"
    data_dir: Path = Path("data")    # notes.md, memory.json
    vision: bool = True              # describe_screen / click_on via the LLM's image input
    vision_model: str = ""           # "" = same model as llm.model (Qwen3.5 sees images)
    router: bool = True              # fast regex commands skip the LLM
    allow_power: bool = True         # shutdown / restart (always asks for confirmation)
    max_steps: int = 4               # LLM <-> tool rounds per reply
    home_city: str = ""              # weather tool: city used when none is said
    files: dict = field(default_factory=dict)   # file tools via the Filesystem MCP server (tools/files.py)
    mcp_servers: dict = field(default_factory=dict)   # other MCP servers: name -> settings (tools/mcp_bridge.py)
    mcp_wait_s: float = 15.0         # at startup, wait this long for MCP servers before warming the LLM


@dataclass(frozen=True)
class Config:
    device: str
    gpu_index: int
    paths: Paths
    server: Server = field(default_factory=Server)
    audio: Audio = field(default_factory=Audio)
    vad: Vad = field(default_factory=Vad)
    turn: Turn = field(default_factory=Turn)
    stt: Stt = field(default_factory=Stt)
    llm: Llm = field(default_factory=Llm)
    tts: Tts = field(default_factory=Tts)
    chunker: Chunker = field(default_factory=Chunker)
    barge_in: BargeIn = field(default_factory=BargeIn)
    tools: Tools = field(default_factory=Tools)


def _resolve(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG, root: Path = PROJECT_ROOT) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    device = raw.get("device", "cuda")
    if device not in ("cuda", "cpu"):
        raise ValueError(f"device must be 'cuda' or 'cpu', got {device!r}")

    paths_raw = raw.get("paths", {})
    paths = Paths(
        models_dir=_resolve(paths_raw.get("models_dir", "models"), root),
        logs_dir=_resolve(paths_raw.get("logs_dir", "logs"), root),
    )
    return Config(
        device=device,
        gpu_index=int(raw.get("gpu_index", 0)),
        paths=paths,
        server=Server(**raw.get("server", {})),
        audio=Audio(**raw.get("audio", {})),
        vad=Vad(**raw.get("vad", {})),
        turn=Turn(**raw.get("turn", {})),
        stt=Stt(**raw.get("stt", {})),
        llm=Llm(**raw.get("llm", {})),
        tts=_tts(raw.get("tts", {}), root),
        chunker=Chunker(**raw.get("chunker", {})),
        barge_in=BargeIn(**raw.get("barge_in", {})),
        tools=_tools(raw.get("tools", {}), root),
    )


def _tools(d: dict, root: Path) -> Tools:
    d = dict(d)
    if "screenshots_dir" in d:
        d["screenshots_dir"] = _resolve(Path(str(d["screenshots_dir"])).expanduser(), root)
    d["data_dir"] = _resolve(d.get("data_dir", "data"), root)
    return Tools(**d)


def _tts(d: dict, root: Path) -> Tts:
    # This function reads the tts section and checks the voice settings.
    t = Tts(**d)
    if t.language not in ("en", "ar") or t.gender not in ("female", "male"):
        raise ValueError(f"tts: language en|ar, gender female|male (got {t.language}, {t.gender})")
    return t
