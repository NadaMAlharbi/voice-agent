# Voice Agent (HAZEM)

A local, real-time voice assistant for **Arabic and English** that runs on your own Windows PC
with an NVIDIA GPU. You talk, it answers with a voice, and you can interrupt it at any time.
It can also control the computer: open and close apps, switch windows, type text, search the web,
take screenshots, change the volume, set timers, and more.

## Why this project

Most voice assistants send your voice to the cloud and handle Arabic poorly. This project keeps
speech recognition, the language model and speech synthesis on your machine, supports Arabic and
English in the same conversation, and keeps the reply delay low by streaming every step.

## Main features

| Feature | What it does |
|---|---|
| Local voice loop | Microphone -> speech to text -> LLM -> text to speech, all on your PC |
| Arabic and English | Each reply uses the language you just spoke. No manual switch needed |
| Female and male voices | Four voices: English female/male, Arabic female/male |
| Barge-in | Speak while it talks and it stops at once. Its own echo and short noises are ignored |
| Fast commands | Clear commands ("open notepad", "كم الساعة") run without the LLM |
| Computer control | Apps, windows, keyboard, mouse, screenshots, volume, media keys, power |
| Safety confirmations | Risky actions (closing apps, shutdown, some shortcuts) ask "yes or no" first |
| Screen vision | "What's on my screen?" and "click the Sign in button" use the same local LLM |
| Timers, notes, memory | Spoken reminders, notes in `data/notes.md`, facts in `data/memory.json` |
| File tools (MCP) | List, find, read, write, edit, move and delete files in `Documents/HAZEM` only |
| Sleep mode | "Go to sleep" / "نام" mutes it until you say "HAZEM" / "حزم" |
| Browser UI | SAMI Navantia design: radar dish that shows the state, live captions, settings (language, voice, captions), activity panel with typed commands |

## How it works

```
Microphone (browser, 16 kHz)
  -> VAD + end of turn      Silero VAD + Smart Turn v3.2       turn.py, server/bargein.py
  -> Speech to text         faster-whisper large-v3-turbo      stt/
  -> Agent                  sleep/wake, yes/no, fast router    server/agent.py, tools/router.py
  -> Tool or LLM            tools/*.py or Qwen3.5 (Ollama)     tools/, llm.py
  -> Chunker                reply text -> short chunks         chunker.py, server/tts_budget.py
  -> Text to speech         Kokoro (EN) / XTTS-v2 (AR)         tts/
  -> Speaker (browser, 24 kHz audio over the WebSocket)
```

One turn, step by step:

1. The browser sends microphone audio to the server over a WebSocket.
2. VAD finds speech; Smart Turn decides when you have finished talking.
3. Whisper turns the speech into text (Arabic or English only).
4. The router checks for simple commands. A match runs the tool and speaks a short fixed sentence.
5. Otherwise the LLM answers and may call tools. Its text is cut into short chunks as it streams.
6. Each chunk is spoken as soon as it is ready, while the next one is being made.
7. If you speak while it talks, the audio stops and your new request is handled.

Speed tricks: when Smart Turn is not sure you finished, the LLM already starts on the words
heard so far (its output is kept only if you really stopped). If a sentence was cut at a
pause and you go on talking, both parts are joined into one request.

A simple Arabic guide to the files, terms and the request path: `PROJECT_GUIDE.md`.

## Project structure

```
voice-agent/
  config/
    config.yaml            all settings (models, voices, chunk sizes, tools, server)
    models.yaml            where each model is downloaded from
  src/voice_agent/
    config.py              loads config.yaml
    models.py              download-models / verify-models, offline mode
    check_env.py           check-env (Python, driver, GPU, CUDA libraries)
    turn.py                VAD + end of turn (also the vad-test command)
    stt/                   speech to text (Whisper), stt-test, word error rate
    llm.py                 Ollama client (streaming, tool calls), llm-test
    knowledge.py           time/date from the clock, "no live data" note for news, prices, etc.
    language.py            reply language and a filter that removes other scripts
    chunker.py             streaming text -> speakable chunks, chunk-test
    tts/
      base.py              voices from config, picks the engine per voice
      kokoro_engine.py     English voices (Kokoro-82M, ONNX)
      xtts_engine.py       Arabic voices (XTTS-v2, streaming)
      tone_engine.py       test beeps, no model
      cli.py               tts-test
    server/
      app.py               FastAPI server, the serve command
      pipelines.py         builds the pipeline for each mode
      agent.py             router first, then LLM with tools; confirmations
      response.py          one reply: LLM stream, chunks, TTS, barge-in, timers
      bargein.py           decides if sound during speech is you or its own echo
      tts_budget.py        splits chunks so audio has no gaps
      processors.py, serializer.py   Pipecat helpers and the WebSocket format
      probe.py             ws-probe test client
      static/              browser UI (index.html, app.js, radar.js, audio-worklets.js, turn-gate.js, styles.css, logo.png)
    tools/
      toolbox.py           collects all tools (FEATURES list)
      files.py             file tools (list/read/write/edit/move/delete) on the Filesystem MCP server
      file_access.py       the one allowed folder + its MCP server process
      mcp_client.py        small MCP client (stdio JSON-RPC)
      mcp_bridge.py        MCP servers from config.yaml -> tools (<server>__<tool>)
      mcp_cli.py           mcp-test: list / call MCP tools without the voice loop
      router.py            fast English/Arabic command patterns
      speech.py            sentences spoken after router actions and confirmations
      desktop.py           Windows / X11 / dry-run backends
      win_input.py         layout-independent keyboard input on Windows
      apps.py              spoken app names (EN/AR) -> installed apps
      calculator.py        calculations in the Windows Calculator
      launch_app.py, open_window.py, gui.py, screenshot.py, screen_vision.py, vision.py,
      website_open.py, google_search.py, youtube_search.py, google_calendar.py,
      media_volume.py, power.py, system_stats.py, date_time.py, weather.py,
      timers.py, notes.py, memory.py
      cli.py               tools-test and agent-test
    bench.py               latency-bench
    e2e_bench.py           latency-bench --e2e: end of speech -> first audio
  scripts/make_bench_clips.py   makes tests/audio/bench/ (synthetic EN + AR clips)
    record.py              record-clip
  tests/                   pytest tests; tests/audio/ has short speech clips
  pyproject.toml           dependencies and commands
```

Created at run time (ignored by git): `models/`, `logs/`, `data/`, `outputs/`, `recordings/`, `.venv/`.

## Technologies and models

| Part | Model / library | Notes |
|---|---|---|
| Voice pipeline | Pipecat (`pipecat-ai`) | frames, VAD and turn analyzers |
| VAD | Silero VAD (bundled with Pipecat) | detects speech |
| End of turn | Smart Turn v3.2 (bundled with Pipecat) | decides when you finished a sentence |
| Speech to text | faster-whisper `large-v3-turbo` (GPU, float16) | language limited to Arabic/English |
| LLM | `qwen3.5:9b` in Ollama, thinking off | replies, tool calls, screen images |
| English TTS | Kokoro-82M (`kokoro-onnx`) | voices `af_heart` (female), `am_michael` (male) |
| Arabic TTS | XTTS-v2 (`coqui-tts`, GPU) | speakers `Ana Florence` (female), `Damien Black` (male) |
| Web server | FastAPI + Uvicorn + WebSocket | UI at http://127.0.0.1:8000 |
| Computer control | ctypes (Win32), pyautogui, uiautomation, mss, pyperclip, psutil, Pillow | Windows |

License note: XTTS-v2 is under the Coqui Public Model License (non-commercial use).
The XTTS speakers are not native Arabic speakers, so a slight accent is expected.

## System requirements

| Item | Requirement |
|---|---|
| OS | Native Windows (the computer-control tools use the Win32 API) |
| GPU | NVIDIA GPU; driver 570 or newer (CUDA 12.8 PyTorch builds) |
| Python | 3.12 (installed by `uv` from `.python-version`) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| LLM server | [Ollama](https://ollama.com) with `qwen3.5:9b` pulled |
| Browser | Chrome or Edge (microphone access, AudioWorklet) |
| Node.js (optional) | Only for the file tools (Filesystem MCP server) |
| Audio | Microphone, and headphones or speakers |
| Internet | Only for setup (packages, models) and for web-based tools (see below) |

`check-env` shows a warning on WSL2 or Linux: the project is built for native Windows.
If you start the server inside WSL2, the desktop tools do not control your Windows apps.

## Installation (Windows PowerShell)

```powershell
# 1. Install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. In the project folder: install Python 3.12 and all packages into .venv
uv sync

# 3. Download the speech models into models/ (needs internet, once)
uv run download-models

# 4. Install Ollama, then download the LLM
ollama pull qwen3.5:9b

# 5. Check the setup
uv run check-env
uv run verify-models
uv run pytest -q
```

After this, the assistant itself works without internet.

## Configuration

Everything is in `config/config.yaml`.

| What | Setting |
|---|---|
| GPU or CPU | `device`, `gpu_index` |
| Server address | `server.host`, `server.port` |
| Speech detection | `vad.*`, `turn.*` |
| Speech to text | `stt.language` (`auto`, `ar`, `en`), `stt.whisper.*` |
| LLM | `llm.model`, `llm.temperature`, `llm.max_tokens`, `llm.num_ctx`, `llm.system_prompt` |
| Default voice | `tts.language` (`en`/`ar`), `tts.gender` (`female`/`male`) |
| Voice list | `tts.voices` |
| TTS speed | `tts.kokoro.speed`, `tts.xtts.speed` |
| XTTS streaming | `tts.xtts.stream_chunk_tokens`, `tts.xtts.temperature` |
| Chunk sizes (speed vs. smoothness) | `chunker.*` |
| Barge-in sensitivity | `barge_in.*` |
| Tools | `tools.enabled`, `tools.router`, `tools.desktop`, `tools.browser`, `tools.vision`, `tools.allow_power`, `tools.home_city`, `tools.max_steps` |
| File tools (MCP) | `tools.files.enabled`, `tools.files.root`, `tools.files.command` |
| Far voices / quiet voice | `vad.min_volume`, `barge_in.min_volume` |
| Model download sources | `config/models.yaml` |

## Running

```powershell
uv run serve --mode assistant
```

Then open http://127.0.0.1:8000 in Chrome or Edge and allow the microphone.

Note: without `--mode`, `serve` starts in `transcribe` mode (speech to text only).

| Mode | What it does |
|---|---|
| `loopback` | Plays your microphone back to you (audio check) |
| `transcribe` | Live transcripts and turn detection only (default) |
| `chat` | LLM replies as text, no speech |
| `assistant` | Full voice assistant with tools and barge-in |

Other `serve` options: `--host`, `--port`, `--debug` (Pipecat debug logs),
`--record-segments` (saves every STT segment to `logs/segments/` as wav + json).

## Using the assistant

1. Click the green microphone button (or the radar) to start. The radar and the label show the
   state: ready, listening, thinking, speaking, asleep.
2. Speak normally. Your words and the reply appear as large captions. You can interrupt at any time.
3. **Stop** ends the current reply at once (the server stops the LLM and the speech).
4. The settings button (sliders icon) opens Language (Auto / Arabic / English), Voice
   (Female / Male) and Captions. The language choice is kept in the browser.
5. The activity button (pulse icon) opens the typed command box, the conversation, timings
   and the log (transcripts, tool calls, "voice -NN LUFS").
6. Yes / No buttons appear next to the captions when an action needs confirmation.
7. White theme: open http://127.0.0.1:8000/?theme=white

### Arabic and English

- No switch is needed. Each reply uses the language of what you just said.
- `tts.language` is only the default before you have spoken (for example, for timer announcements).
- To force speech recognition to one language, pick Arabic or English in Settings
  (or set the default `stt.language: ar` or `en`).

### Female and male voices

- Use the **Female / Male** buttons in the browser. The change applies from the next spoken chunk.
- To change the default, set `tts.gender` in `config.yaml`.
- To use other voices, edit `tts.voices`. Kokoro American voices start with `af_` (female) or
  `am_` (male). XTTS-v2 has many built-in speakers; if a name is wrong, the error lists the valid ones.
- Each voice must be different: loading fails if two entries use the same model voice.

### Sleep and wake

- "Go to sleep", "stop listening", "نام", "وضع السكون": the assistant ignores speech.
- "HAZEM", "wake up", "حزم", "استيقظ": it wakes up ("HAZEM, open notepad" also works).

## Computer-control tools

Each tool is a small file in `src/voice_agent/tools/`.

| Tool | What it does | Asks first? |
|---|---|---|
| `open_app` | Open an installed app or a known website | No |
| `close_app` | Close an app's windows (unsaved changes are discarded) | **Yes** |
| `list_windows`, `focus_window`, `window_action` | List, switch to, minimize/maximize/restore windows | No |
| `type_text` | Type text into the app you were using | No |
| `press_keys` | Press a key or shortcut | **Yes** for alt+f4, shift+delete, ctrl+shift+delete, ctrl+shift+q, ctrl+shift+w |
| `move_mouse`, `mouse_click`, `scroll`, `read_clipboard` | Mouse and clipboard | No |
| `take_screenshot` | Save a screenshot to `~/Pictures/HAZEM` | No |
| `describe_screen`, `click_on` | Describe the screen, click an element by name (vision) | No |
| `open_url`, `web_search`, `youtube_search` | Open a website, search Google / Maps / Wikipedia / images / YouTube | No |
| `open_calendar`, `add_calendar_event` | Open Google Calendar, or open a pre-filled new event (you press Save) | No |
| `volume`, `media` | Volume up/down/mute, play/pause, next, previous | No |
| `system_status` | CPU, memory, disk, battery, GPU | No |
| `lock_screen` | Lock the computer | No |
| `power` | Shut down or restart after 30 s, or cancel | **Yes** (cancel does not ask) |
| `get_date_time`, `get_weather` | Clock; current weather (internet) | No |
| `calculate` | Types the calculation into Windows Calculator (you see the keys), reads the result | No |
| `list_files`, `read_file` | List or find files, read a text file (only in `Documents/HAZEM`) | No |
| `write_file`, `edit_file`, `move_file`, `delete_file` | Change files (only in `Documents/HAZEM`; delete moves to `.deleted`) | **Yes** |
| `set_timer`, `add_note`, `read_notes`, `remember`, `forget` | Timers, notes, memory | No |

How confirmations work:

- The assistant asks, for example, "Should I close Notepad? Unsaved changes will be lost. Say yes or no."
- Answer by voice ("yes", "نعم", "no", "لا") or with the Yes / No buttons in the browser.
- Any other reply drops the action and is handled as a normal request. A question expires after 45 seconds.
- `tools.allow_power: false` blocks shutdown and restart completely.

Typing notes:

- Text goes to the last app you used, not to the HAZEM browser tab.
- All text (Arabic, English, numbers, punctuation, mixed) is pasted through the clipboard as exact
  Unicode, so no character is dropped or auto-corrected. Your old clipboard content is put back
  about 1.5 s later.
- If the target app runs as administrator, Windows blocks the keys unless HAZEM also runs as administrator.

Calendar notes: the calendar tools only open web pages in the browser. There is no Google account
connection, so the assistant cannot read, cancel, delete or decline events.

### Adding a tool

1. Create `src/voice_agent/tools/<feature>.py` with handler functions and a `tools(tb)` function that
   returns `Tool(...)` entries (see `youtube_search.py` for a short example).
2. Import the module in `tools/toolbox.py` and add it to `FEATURES`.
3. Optional: add a spoken sentence in `tools/speech.py` and a fast pattern in `tools/router.py`.
4. For a risky tool, pass `risk=CONFIRM` to `Tool(...)`.
5. Add a test in `tests/test_tools.py` (the `box` fixture uses the dry-run desktop).

## Tests and check commands

```powershell
uv run pytest -q                       # all unit and end-to-end tests (no GPU needed for most)
uv run pytest tests/test_tools.py      # tools, router, confirmations (dry-run desktop)
uv run pytest tests/test_tts.py        # voices and TTS engines
```

Some tests are skipped automatically: `test_frontend_js.py` and the server part of `test_files.py`
need Node.js, one Kokoro test needs
the downloaded model, and two tests need PyTorch.

| Command | What it checks |
|---|---|
| `uv run check-env` | Python, platform, NVIDIA driver, PyTorch CUDA, GPU kernels, CTranslate2, ONNX Runtime (Kokoro) GPU |
| `uv run verify-models` | All models are on disk (offline check) |
| `uv run record-clip NAME --seconds 5` | Records a mic clip to `recordings/NAME.wav` (`--list-devices` to list mics) |
| `uv run vad-test FILE.wav` | Speech and end-of-turn detection on a file |
| `uv run stt-test FILE.wav` | Transcribes files; `--self-check` checks that noise gives no text |
| `uv run llm-test` | LLM replies; `--lang-check` and `--realtime-check` for language and live-data behavior |
| `uv run chunk-test "text"` | Shows how a reply is split into chunks; `--llm "prompt"` uses a real reply |
| `uv run tts-test` | Speaks a sample with every voice, saves `outputs/tts/*.wav`; `--asr` adds a Whisper check |
| `uv run latency-bench` | Simulated first-audio time and gaps; `--real` uses Ollama and the real TTS |
| `uv run latency-bench --e2e` | End of speech -> first audio, p50/p90, through the real VAD + Smart Turn (model times simulated) |
| `uv run latency-bench --e2e --url ws://127.0.0.1:8000/ws` | The same with your running server and real models |
| `uv run llm-test --cache-check` | Checks that Ollama reuses the cached system prompt + tools |
| `uv run tools-test` | Runs the desktop tools for real: opens Notepad, types EN + AR, then closes it |
| `uv run agent-test` | 26 EN/AR commands through router + LLM (needs Ollama); nothing is executed. `--llm-only` skips the router |
| `uv run ws-probe tests\audio\en_question.wav` | Streams a wav to a running server and prints the events |
| `uv run mcp-test` | Starts the MCP servers in `tools.mcp_servers`, lists their tools. `uv run mcp-test calculator add_numbers a=2 b=3` calls one |

Do not touch the mouse or keyboard while `tools-test` runs. Move the mouse to a screen corner to abort.

## File tools (MCP)

The file tools use the official Filesystem MCP server (`@modelcontextprotocol/server-filesystem`).
It runs on your PC as a child process and can only use one folder (`tools.files.root`,
default `Documents\HAZEM`). Nothing is sent over the network.

Setup on Windows:

1. Install Node.js LTS from https://nodejs.org (it brings `npx`). Check: `npx --version`.
2. Start the server once with internet so npm downloads the MCP server:
   `uv run serve --mode assistant`. The log says `tools: file tools ready ...`.
   Later starts work offline (`--prefer-offline` uses the npm cache).
3. Ask: "what files do I have", "read the file plan.txt", "اكتب ملف اسمه مهام فيه ...".
   Writing, editing, moving and deleting ask "yes or no" first. Delete moves the file to
   `Documents\HAZEM\.deleted`, so you can restore it.

Without Node.js the file tools are simply not offered (a warning is logged).
To turn them off: `tools.files.enabled: false`.

## Tools from other projects (MCP)

Any project can add tools by shipping an MCP server: add it under `tools.mcp_servers` in
`config/config.yaml`, no code change. Example included: `examples/mcp-calculator`
(tool `calculator__add_numbers`).

```powershell
uv sync --directory examples/mcp-calculator     # once, needs internet
uv run mcp-test calculator add_numbers a=2 b=3
```

Guide: [docs/MCP_GUIDE.md](docs/MCP_GUIDE.md). For other developers:
[docs/MCP_INTEGRATION_CHECKLIST.md](docs/MCP_INTEGRATION_CHECKLIST.md).

## Offline use

| Works offline | Needs internet |
|---|---|
| Speech to text, LLM, TTS, barge-in | First setup: `uv sync`, `download-models`, `ollama pull` |
| Time, date, timers, notes, memory | `get_weather` (Open-Meteo) |
| Apps, windows, keyboard, mouse, screenshots, volume | Web search, YouTube, websites, Google Calendar (they open in the browser) |
| Screen description (local LLM) | |

The server sets Hugging Face offline mode at runtime, so models are never downloaded during use.
The LLM is told it has no live data, so it should not invent news, prices or scores.

## Troubleshooting

| Problem | What to try |
|---|---|
| `check-env`: "Kernel support" FAIL | PyTorch is too old for your GPU. Run `uv sync` again (CUDA 12.8 builds). |
| `check-env`: "CUDA build" FAIL | A CPU-only PyTorch is installed. Run `uv sync` in the project folder. |
| `verify-models` shows missing files | Run `uv run download-models` with internet. |
| `Kokoro has no voice ...` / `XTTS has no speaker ...` | A name in `tts.voices` is wrong; the error lists valid names. |
| "LLM not reachable" in the log | Start the Ollama app or run `ollama serve`, and check `llm.host`. |
| The page loads but nothing answers | Start the server with `--mode assistant`. |
| The assistant interrupts itself | Lower the speaker volume or use headphones. |
| It does not hear quiet speech | Look at "voice -NN LUFS" in the activity log. Lower `vad.min_volume` (volume = (LUFS + 110) / 100, minus about 0.05), or open the page with `?agc=1`. |
| It reacts to people far away or the TV | Raise `vad.min_volume` and `barge_in.min_volume` a little (for example 0.75 / 0.77). A headset helps most. NVIDIA Broadcast noise removal also works. |
| It answers before I finish a sentence | Raise `turn.stop_secs` (for example 1.5). |
| `kokoro-82M onnx (cpu) WARNING` in the log | Run `uv run check-env`; see the "ONNX Runtime" and "Kokoro GPU" lines for the fix. |
| `cublasLt64_13.dll ... missing` (Kokoro on CPU) | onnxruntime-gpu 1.27+ on PyPI is built for CUDA 13; torch here is CUDA 12.8. `pyproject.toml` pins `onnxruntime-gpu<1.27`: run `uv sync --reinstall-package onnxruntime-gpu`. |
| `tools: file server failed to start (...)` | The message now ends with what the MCP server printed (`Server said: ...`). Old builds showed only `[Errno 22] Invalid argument`. |
| Arabic voice has an accent | Try other XTTS speakers in `tts.voices` and compare with `uv run tts-test`. |
| Typing goes to the wrong window | Click the target app once before the command; check the activity log for `into ...`. |
| "Windows accepted only N of M key events" | The target app runs as administrator. Run HAZEM as administrator too. |
| Desktop tools do nothing | Check the startup log line `tools: desktop=...`. It must say `windows`. |
| Weather says it is offline | Weather needs internet. Set `tools.home_city` for questions without a city. |

## Limitations

- Only Arabic and English. Arabic replies are in Modern Standard Arabic.
- Arabic speech uses non-native XTTS speakers and can sound less smooth than English.
- Built for native Windows. On Linux/X11 the tools work only partly; in dry-run mode nothing is touched.
- One XTTS/Kokoro synthesis runs at a time on the GPU.
- Google Calendar support is limited to opening the calendar and a pre-filled new event.
- No live data (news, prices, sports) except the weather tool.
- XTTS-v2 is for non-commercial use only.
