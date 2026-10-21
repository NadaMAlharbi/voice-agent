# HAZEM Voice Agent

A local, real-time voice assistant for **Arabic and English** that runs on a Windows PC with an NVIDIA GPU. You speak in the browser, HAZEM answers with a voice, and you can interrupt it at any time. It can also control the computer: open and close apps, switch windows, type text, search the web, take screenshots, change the volume, set timers, and work with files.

Speech recognition, the language model (through **Ollama**), and speech synthesis all run on your machine. After the first setup, the voice loop works without internet.

---

## 1. Features

- **Local voice loop.** Microphone → speech-to-text → LLM → text-to-speech, all on your PC.
- **Arabic and English.** Each reply uses the language you just spoke. No switch is needed.
- **Four voices.** English and Arabic, female and male, switchable from the browser.
- **Barge-in.** Speak while HAZEM talks and it stops at once. Its own echo and short noises are ignored.
- **Fast commands.** Clear commands ("open notepad", "كم الساعة") run without the LLM.
- **Computer control.** Apps, windows, keyboard, mouse, screenshots, volume, media keys, and power.
- **Safety confirmations.** Risky actions (closing apps, shutdown, file changes) ask "yes or no" first.
- **Screen vision.** "What's on my screen?" and "click the Sign in button" use the local LLM.
- **Timers, notes, and memory.** Spoken timers, notes in `data/notes.md`, facts in `data/memory.json`.
- **File tools.** Read and change files, limited to the `Documents/HAZEM` folder.
- **MCP tools.** Tools from other projects can be added in `config.yaml` with no code change.
- **Sleep mode.** "Go to sleep" / "نام" mutes HAZEM until you say "HAZEM" / "حزم".

### How it works

```
Browser mic (16 kHz) → WebSocket /ws
  → Silero VAD + Smart Turn      detects speech and the end of your turn
  → faster-whisper               speech-to-text (Arabic / English)
  → Router or LLM (Ollama)       fast commands, or qwen3.5:9b with tools
  → Chunker                      reply → short chunks, spoken as they stream
  → Kokoro (EN) / XTTS-v2 (AR)   text-to-speech
→ Browser speaker (24 kHz)
```

---

## 2. Requirements

| Item            | Requirement                                                              |
| --------------- | ------------------------------------------------------------------------ |
| OS              | Native Windows (the desktop tools use the Win32 API)                     |
| GPU             | NVIDIA GPU, driver 570 or newer                                          |
| Python          | 3.12 (installed by `uv` from `.python-version`)                          |
| Package manager | [uv](https://docs.astral.sh/uv/)                                         |
| LLM server      | [Ollama](https://ollama.com) with `qwen3.5:9b`                           |
| Browser         | Chrome or Edge                                                           |
| Audio           | A microphone, and headphones or speakers                                 |
| Optional        | Node.js LTS, only for the file tools                                     |

`uv.lock` supports Windows only, so run every command in **Windows PowerShell**, not WSL2.

---

## 3. Installation

In the project folder:

```powershell
# 1. Install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install Python 3.12 and all packages
uv sync

# 3. Download the speech models into models/ (needs internet, once)
uv run download-models

# 4. Download the LLM (after installing Ollama)
ollama pull qwen3.5:9b

# 5. Check the setup
uv run check-env
uv run verify-models
```

**File tools (optional).** Install Node.js LTS from https://nodejs.org and check it with `npx --version`. The first `serve` start needs internet to download the Filesystem MCP server; later starts work offline.

---

## 4. Running

```powershell
uv run serve --mode assistant
```

Open http://127.0.0.1:8000 in Chrome or Edge and allow the microphone. Wait for `ready: mode=assistant` in the log.

| Mode         | What it does                                       |
| ------------ | -------------------------------------------------- |
| `assistant`  | Full voice assistant with tools and barge-in       |
| `chat`       | LLM replies as text, no speech                     |
| `transcribe` | Live transcripts only (default without `--mode`)   |
| `loopback`   | Plays your microphone back (audio check)           |

Other options: `--host`, `--port`, `--debug`, `--record-segments` (saves STT segments to `logs/segments/`).

### Using the browser UI

- Click the microphone button (or the radar) to start, then speak normally. **Stop** ends a reply at once.
- **Settings** (sliders icon): Language (Auto / Arabic / English), Voice (Female / Male), Captions.
- **Activity** (pulse icon): typed commands, conversation, timings, and the log.
- Answer confirmations by voice ("yes", "نعم", "no", "لا") or with the Yes / No buttons. A question expires after 45 seconds.
- White theme: http://127.0.0.1:8000/?theme=white

---

## 5. Configuration

All settings are in `config/config.yaml`. Restart `serve` after a change.

| Setting                          | Default             | Purpose                                                    |
| -------------------------------- | ------------------- | ---------------------------------------------------------- |
| `device`                         | `cuda`              | `cuda` or `cpu`                                            |
| `server.host`, `server.port`     | `127.0.0.1`, `8000` | Server address                                             |
| `stt.language`                   | `auto`              | `auto`, `ar`, or `en`                                      |
| `llm.model`                      | `qwen3.5:9b`        | Ollama model                                               |
| `tts.language`, `tts.gender`     | `en`, `female`      | Default voice before you speak                             |
| `tts.voices`                     | 4 voices            | Model voice for each language and gender                   |
| `vad.min_volume`                 | `0.72`              | Higher ignores far voices; lower hears quiet ones          |
| `barge_in.min_volume`            | `0.74`              | Quieter voices cannot interrupt a reply                    |
| `turn.stop_secs`                 | `1.0`               | Raise if HAZEM answers before you finish                   |
| `tools.router`                   | `true`              | Fast commands without the LLM                              |
| `tools.allow_power`              | `true`              | `false` blocks shutdown and restart                        |
| `tools.home_city`                | `""`                | Weather city when you do not say one                       |
| `tools.files.enabled`, `.root`   | `true`, `~/Documents/HAZEM` | File tools and their only folder                   |
| `tools.mcp_servers`              | `calculator`        | MCP servers from other projects ([Section 7](#7-mcp-tools-from-other-projects)) |

**Tuning voice detection.** The activity log shows your level as `voice -NN LUFS`. The volume scale is `(LUFS + 110) / 100`, so `0.72` = -38 LUFS. Set `vad.min_volume` between your normal voice and the sound to ignore, and `barge_in.min_volume` about 0.02 higher.

---

## 6. Tools

HAZEM's built-in tools live in `src/voice_agent/tools/` (one file per feature, collected in `tools/toolbox.py`).

| Group            | Tools                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------- |
| Apps and windows | `open_app`, `close_app`*, `list_windows`, `focus_window`, `window_action`             |
| Keyboard, mouse  | `type_text`, `press_keys`*, `move_mouse`, `mouse_click`, `scroll`, `read_clipboard`   |
| Screen           | `take_screenshot`, `describe_screen`, `click_on`                                      |
| Web              | `open_url`, `web_search`, `youtube_search`, `open_calendar`, `add_calendar_event`     |
| System           | `volume`, `media`, `system_status`, `lock_screen`, `power`*                           |
| Info             | `get_date_time`, `get_weather` (needs internet), `calculate` (Windows Calculator)     |
| Files            | `list_files`, `read_file`, `write_file`*, `edit_file`*, `move_file`*, `delete_file`*  |
| Personal         | `set_timer`, `add_note`, `read_notes`, `remember`, `forget`                           |

\* Asks "yes or no" first (`press_keys` only for dangerous shortcuts such as alt+f4; `power` not for cancel).

Notes: text is typed into the last app you used, not the HAZEM tab. The calendar tools only open pages in the browser; there is no Google account connection.

---

## 7. MCP tools from other projects

HAZEM can use tools from any **MCP server** (stdio). HAZEM starts the server as a child process and shows its tools to the LLM as `<server>__<tool>`. Adding a server needs only an entry in `config/config.yaml`:

```yaml
tools:
  mcp_servers:
    calculator:
      enabled: true
      command: [uv, run, --directory, examples/mcp-calculator, mcp-calculator]
      tools: [add_numbers]
      confirm: []
      safe: []
      timeout_s: 30
      start_timeout_s: 120
      hint: "To add two numbers with the MCP calculator, use calculator__add_numbers."
```

Try the included example:

```powershell
uv sync --directory examples/mcp-calculator     # once, needs internet
uv run mcp-test                                  # list servers and tools
uv run mcp-test calculator add_numbers a=2 b=3   # call a tool
```

Then say: "Add 25 and 17 with the MCP calculator."

> **Note:** the fast router runs before the LLM and only knows built-in tools. For example, "save a note: ..." always uses the built-in `add_note`, even if an MCP server also has a notes tool. Name the MCP server in the request to reach its tools.

**To build and connect your own server** (with a complete MCP Notes example), see **[MCP_GUIDE.md](MCP_GUIDE.md)**.

---

## 8. Check commands

| Command                  | What it does                                                         |
| ------------------------ | -------------------------------------------------------------------- |
| `uv run check-env`       | Checks Python, the NVIDIA driver, PyTorch CUDA, and ONNX Runtime GPU |
| `uv run verify-models`   | Checks that all models are on disk                                   |
| `uv run mcp-test`        | Starts the MCP servers and lists or calls their tools                |

More diagnostic commands are defined under `[project.scripts]` in `pyproject.toml`.

---

## 9. Troubleshooting

| Problem                                    | What to try                                                                  |
| ------------------------------------------ | ---------------------------------------------------------------------------- |
| `uv sync`: not compatible with the lockfile | Run it in Windows PowerShell, not WSL2.                                     |
| `check-env`: "CUDA build" or "Kernel support" FAIL | Run `uv sync` again in the project folder (CUDA 12.8 PyTorch).        |
| `verify-models` shows missing files        | Run `uv run download-models` with internet.                                  |
| "LLM not reachable" in the log             | Start Ollama (`ollama serve`) and check `llm.host`.                          |
| The page loads but nothing answers         | Start with `--mode assistant`.                                               |
| HAZEM interrupts itself                    | Lower the speaker volume or use headphones.                                  |
| It reacts to far voices or the TV          | Raise `vad.min_volume` and `barge_in.min_volume` a little (for example 0.75 / 0.77). |
| It does not hear quiet speech              | Lower `vad.min_volume`, or open the page with `?agc=1`.                      |
| Desktop tools do nothing                   | The startup log must say `tools: desktop=windows`.                           |
| `tools: MCP <name> failed to start`        | Run `uv run mcp-test <name>` and see [MCP_GUIDE.md](MCP_GUIDE.md#8-troubleshooting). |

---

## 10. Limitations

- Arabic and English only. Arabic voices use non-native XTTS speakers, so a slight accent is expected.
- No live data (news, prices, sports) except the weather tool.
- XTTS-v2 is under the Coqui Public Model License (non-commercial use only).

---

## 11. Documentation

| Guide                                                          | For                                                         |
| -------------------------------------------------------------- | ----------------------------------------------------------- |
| [MCP_GUIDE.md](MCP_GUIDE.md)                                   | Creating an MCP server, connecting it, configuring and testing it |
| [examples/mcp-calculator/README.md](examples/mcp-calculator/README.md) | The smallest MCP server, used as a template          |
| `config/config.yaml`                                           | Every setting, with comments                                |
