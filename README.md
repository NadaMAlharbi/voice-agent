# HAZEM

HAZEM is a local, real-time voice assistant for **Arabic and English**. It runs on your own
Windows PC with an NVIDIA GPU. You talk, HAZEM answers with a voice, and you can interrupt it
at any time. HAZEM can also control the computer: open apps, switch windows, type text, search
the web, take screenshots, change the volume, set timers, and more.

Speech recognition, the language model and speech synthesis all run on your machine.

## Features

| Feature | What it does |
|---|---|
| Local voice loop | Microphone -> speech to text -> LLM -> text to speech, all on your PC |
| Arabic and English | Each reply uses the language you just spoke |
| Four voices | English female/male, Arabic female/male |
| Barge-in | Speak while HAZEM talks and it stops at once |
| Fast commands | Clear commands ("open notepad", "كم الساعة") run without the LLM |
| Computer control | Apps, windows, keyboard, mouse, screenshots, volume, media keys, power |
| Confirmations | Risky actions (closing apps, shutdown, deleting files) ask "yes or no" first |
| Screen vision | "What's on my screen?" and "click the Sign in button" |
| Timers, notes, memory | Spoken reminders, notes in `data/notes.md`, facts in `data/memory.json` |
| File tools | Work with files in `Documents\HAZEM` only |
| Tools from other projects | Any MCP server can add tools to HAZEM with a few lines of config |
| Sleep mode | "Go to sleep" / "نام" mutes HAZEM until you say "HAZEM" / "حزم" |
| Browser UI | Radar that shows the state, live captions, settings, activity panel |

## How it works

```
Microphone (browser)
  -> Speech detection + end of turn    Silero VAD + Smart Turn v3.2
  -> Speech to text                    faster-whisper large-v3-turbo
  -> Fast router or LLM                tools/router.py or Qwen3.5 (Ollama)
  -> Tools                             tools/*.py and MCP servers
  -> Text to speech                    Kokoro (English) / XTTS-v2 (Arabic)
  -> Speaker (browser)
```

1. The browser streams microphone audio to the server over a WebSocket.
2. HAZEM detects when you finish talking and turns your speech into text.
3. Simple commands run directly. Everything else goes to the LLM, which may call tools.
4. The reply is cut into short chunks and each chunk is spoken as soon as it is ready.
5. If you speak while HAZEM talks, it stops and handles your new request.

## Requirements

| Item | Requirement |
|---|---|
| OS | Native Windows (not WSL2: the desktop tools use the Win32 API) |
| GPU | NVIDIA GPU, driver 570 or newer |
| Python | 3.12 (installed by `uv`) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| LLM server | [Ollama](https://ollama.com) with `qwen3.5:9b` |
| Browser | Chrome or Edge |
| Node.js (optional) | Only for the file tools |
| Internet | Only for setup and web-based tools (weather, search, websites) |

## Install (Windows PowerShell)

```powershell
# 1. Install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. In the project folder: install Python and all packages
uv sync

# 3. Download the speech models (needs internet, once)
uv run download-models

# 4. Install Ollama, then download the LLM
ollama pull qwen3.5:9b

# 5. Check the setup
uv run check-env
uv run verify-models
```

## Run

```powershell
uv run serve --mode assistant
```

Open http://127.0.0.1:8000 in Chrome or Edge and allow the microphone.

| Mode | What it does |
|---|---|
| `assistant` | Full voice assistant with tools (use this) |
| `chat` | LLM replies as text, no speech |
| `transcribe` | Speech to text only (the default when `--mode` is not given) |
| `loopback` | Plays your microphone back to you (audio check) |

Other options: `--host`, `--port`, `--debug`, `--record-segments`.

## Use

1. Click the microphone button (or the radar) to start.
2. Speak normally. Your words and the reply appear as captions.
3. **Stop** ends the current reply at once.
4. **Settings** (sliders icon): language (Auto / Arabic / English), voice (Female / Male), captions.
5. **Activity** (pulse icon): typed commands, conversation, timings and the log.
6. When an action needs confirmation, say "yes" / "نعم" or "no" / "لا", or use the buttons.
7. White theme: http://127.0.0.1:8000/?theme=white

Sleep and wake:

| Say | Result |
|---|---|
| "Go to sleep", "stop listening", "نام", "وضع السكون" | HAZEM ignores speech |
| "HAZEM", "wake up", "حزم", "استيقظ" | HAZEM wakes up ("HAZEM, open notepad" also works) |

## Configuration

All settings are in `config/config.yaml`. Model download sources are in `config/models.yaml`.

| What | Setting |
|---|---|
| GPU or CPU | `device`, `gpu_index` |
| Server address | `server.host`, `server.port` |
| Speech detection | `vad.*`, `turn.*` |
| Speech to text | `stt.language` (`auto`, `ar`, `en`), `stt.whisper.*` |
| LLM | `llm.model`, `llm.temperature`, `llm.max_tokens`, `llm.num_ctx` |
| Default voice | `tts.language` (`en`/`ar`), `tts.gender` (`female`/`male`) |
| Voice list and speed | `tts.voices`, `tts.kokoro.speed`, `tts.xtts.speed` |
| Barge-in sensitivity | `barge_in.*` |
| Tools | `tools.enabled`, `tools.router`, `tools.allow_power`, `tools.home_city` |
| File tools | `tools.files.enabled`, `tools.files.root` |
| MCP servers from other projects | `tools.mcp_servers` (see [MCP_GUIDE.md](MCP_GUIDE.md)) |

## Tools

| Tool | What it does | Asks first? |
|---|---|---|
| `open_app`, `close_app` | Open or close an app | Close: **yes** |
| `list_windows`, `focus_window`, `window_action` | List, switch, minimize/maximize windows | No |
| `type_text`, `press_keys` | Type text, press a shortcut | Some shortcuts (alt+f4, ...) |
| `move_mouse`, `mouse_click`, `scroll`, `read_clipboard` | Mouse and clipboard | No |
| `take_screenshot` | Save a screenshot to `~/Pictures/HAZEM` | No |
| `describe_screen`, `click_on` | Describe the screen, click an element by name | No |
| `open_url`, `web_search`, `youtube_search` | Websites and searches (in the browser) | No |
| `open_calendar`, `add_calendar_event` | Open Google Calendar or a pre-filled new event | No |
| `volume`, `media` | Volume and media keys | No |
| `system_status`, `lock_screen` | CPU/memory/disk/GPU, lock the PC | No |
| `power` | Shut down or restart after 30 s | **Yes** |
| `get_date_time`, `get_weather` | Clock, current weather (internet) | No |
| `calculate` | Uses the Windows Calculator | No |
| `list_files`, `read_file` | List or read files in `Documents\HAZEM` | No |
| `write_file`, `edit_file`, `move_file`, `delete_file` | Change files in `Documents\HAZEM` | **Yes** |
| `set_timer`, `add_note`, `read_notes`, `remember`, `forget` | Timers, notes, memory | No |

Notes:

- Text is typed into the last app you used, not into the HAZEM browser tab.
- If the target app runs as administrator, run HAZEM as administrator too.
- `tools.allow_power: false` blocks shutdown and restart.
- Deleted files go to `Documents\HAZEM\.deleted`, so you can restore them.

### Add a built-in tool

1. Create `src/voice_agent/tools/<feature>.py` with a `tools(tb)` function that returns
   `Tool(...)` entries (see `youtube_search.py`).
2. Add the module to `FEATURES` in `tools/toolbox.py`.
3. Optional: a spoken sentence in `tools/speech.py`, a fast pattern in `tools/router.py`.
4. For a risky tool, pass `risk=CONFIRM` to `Tool(...)`.

### File tools

The file tools use the official Filesystem MCP server. It runs on your PC and can only use
one folder (`tools.files.root`, default `Documents\HAZEM`).

1. Install Node.js LTS from https://nodejs.org. Check with `npx --version`.
2. Start HAZEM once with internet so npm downloads the server. The log says `tools: file tools ready`.
3. Ask: "what files do I have", "read the file plan.txt".

Without Node.js the file tools are not offered. To turn them off: `tools.files.enabled: false`.

### Tools from other projects (MCP)

Another project can give HAZEM new tools by shipping an MCP server. You add it to
`config/config.yaml`; no HAZEM code changes. A working example is in `examples/mcp-calculator`.

Step-by-step guide for developers: [MCP_GUIDE.md](MCP_GUIDE.md).

## Check commands

These commands check one part of HAZEM without the full voice loop.

| Command | What it checks |
|---|---|
| `uv run check-env` | Python, GPU driver, CUDA, ONNX Runtime |
| `uv run verify-models` | All models are on disk |
| `uv run record-clip NAME --seconds 5` | Records a mic clip to `recordings/NAME.wav` |
| `uv run vad-test FILE.wav` | Speech and end-of-turn detection on a file |
| `uv run stt-test FILE.wav` | Speech to text on a file |
| `uv run llm-test` | LLM replies |
| `uv run chunk-test "text"` | How a reply is split into spoken chunks |
| `uv run tts-test` | Speaks a sample with every voice, saves `outputs/tts/*.wav` |
| `uv run latency-bench` | Reply delay (first audio, gaps) |
| `uv run tools-test` | Runs the desktop tools for real (opens Notepad, types, closes it) |
| `uv run agent-test` | Checks that EN/AR commands pick the right tool (needs Ollama, nothing runs) |
| `uv run ws-probe FILE.wav` | Streams a wav file to a running server and prints the events |
| `uv run mcp-test` | Starts the MCP servers from `config.yaml` and lists their tools |

Do not touch the mouse or keyboard while `tools-test` runs.

## Offline use

| Works offline | Needs internet |
|---|---|
| Speech to text, LLM, text to speech | First setup (`uv sync`, `download-models`, `ollama pull`) |
| Time, timers, notes, memory | Weather |
| Apps, windows, keyboard, mouse, screenshots, volume | Web search, YouTube, websites, Google Calendar |

## Project structure

```
config/
  config.yaml          all settings
  models.yaml          model download sources
src/voice_agent/
  server/              FastAPI server, pipelines, agent, barge-in, browser UI (static/)
  stt/                 speech to text (Whisper)
  tts/                 text to speech (Kokoro, XTTS-v2)
  tools/               all tools, the fast router, MCP client and bridge
  llm.py               Ollama client
  turn.py              speech detection and end of turn
  chunker.py           reply text -> spoken chunks
  config.py            loads config.yaml
examples/
  mcp-calculator/      example MCP server from another project
MCP_GUIDE.md           how to connect an MCP project to HAZEM
pyproject.toml         dependencies and commands
```

Created at run time (ignored by git): `models/`, `logs/`, `data/`, `outputs/`, `recordings/`, `.venv/`.

## Models

| Part | Model |
|---|---|
| Speech detection | Silero VAD, Smart Turn v3.2 (bundled with Pipecat) |
| Speech to text | faster-whisper `large-v3-turbo` |
| LLM | `qwen3.5:9b` in Ollama |
| English voices | Kokoro-82M (`af_heart`, `am_michael`) |
| Arabic voices | XTTS-v2 (`Ana Florence`, `Damien Black`) |

XTTS-v2 is under the Coqui Public Model License (non-commercial use). Its speakers are not
native Arabic speakers, so a slight accent is expected.

## Troubleshooting

| Problem | What to try |
|---|---|
| `check-env` shows a CUDA or kernel FAIL | Run `uv sync` again in the project folder |
| `verify-models` shows missing files | Run `uv run download-models` with internet |
| "LLM not reachable" in the log | Start Ollama (`ollama serve`) and check `llm.host` |
| The page loads but nothing answers | Start with `--mode assistant` |
| HAZEM interrupts itself | Lower the speaker volume or use headphones |
| HAZEM does not hear quiet speech | Lower `vad.min_volume` (see "voice -NN LUFS" in the activity log) |
| HAZEM reacts to the TV or far voices | Raise `vad.min_volume` and `barge_in.min_volume` a little |
| HAZEM answers before you finish | Raise `turn.stop_secs` (for example 1.5) |
| Kokoro runs on the CPU | Run `uv run check-env` and follow the "ONNX Runtime" line |
| Typing goes to the wrong window | Click the target app once before the command |
| Desktop tools do nothing | The startup log line `tools: desktop=...` must say `windows` |
| An MCP server does not start | Run `uv run mcp-test <name>` and see [MCP_GUIDE.md](MCP_GUIDE.md) |

