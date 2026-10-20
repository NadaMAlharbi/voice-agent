"""FastAPI server = static web UI + one WebSocket per browser tab.

  uv run serve --mode loopback      # hear yourself, check latency
  uv run serve --mode transcribe    # live transcripts + user turns
  uv run serve --mode chat          # + streamed LLM text replies
  uv run serve --mode assistant     # full voice loop with barge-in
  then open http://127.0.0.1:8000

Models load once at startup and are warmed up, so the first reply is as fast as the rest.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from voice_agent.config import Config, load_config
from voice_agent.models import enable_offline
from voice_agent.server.pipelines import MODES, build_task

STATIC = Path(__file__).parent / "static"


def create_app(mode: str, cfg: Config | None = None, stt_engine=None, llm_factory=None,
               tts_engine=None, record_dir=None, live_provider=None, desktop=None, vision=None) -> FastAPI:
    """Pass ready engines in, or leave None to load from config when the mode needs them.

    llm_factory: callable returning a fresh LLM client per connection.
    desktop: tools backend (e.g. DryRunDesktop); None = from config (tools.desktop).
    """
    cfg = cfg or load_config()
    state: dict = {"stt": stt_engine, "llm_factory": llm_factory, "tts": tts_engine, "voices": None,
                   "desktop": desktop, "vision": vision}
    shots = Path(desktop.screenshots_dir if desktop is not None else cfg.tools.screenshots_dir)
    shots.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        enable_offline()
        if mode in ("transcribe", "chat", "assistant") and state["stt"] is None:
            from voice_agent.stt.base import make_engine, warmup

            t0 = time.perf_counter()
            eng = make_engine(cfg.stt.engine, cfg)
            eng.load()
            logger.info(f"STT {eng.name} loaded in {time.perf_counter() - t0:.1f}s, "
                        f"warmup {warmup(eng):.0f} ms | {eng.device_info()}")
            state["stt"] = eng
        if mode in ("chat", "assistant") and state["llm_factory"] is None:
            from voice_agent.llm import OllamaClient

            state["llm_factory"] = lambda: OllamaClient(cfg.llm)
            probe = state["llm_factory"]()
            try:
                ms = await probe.warmup()
                logger.info(f"LLM {cfg.llm.model} warm in {ms:.0f} ms | VRAM share {await probe.gpu_share()}")
            except Exception as e:  # noqa: BLE001
                logger.error(f"LLM not reachable ({e}); is Ollama running? Replies will fail.")
            finally:
                await probe.aclose()
        if mode == "assistant":
            from voice_agent.tts.base import load_voices, make_tts

            state["voices"] = load_voices(cfg)
            if state["tts"] is None:
                t0 = time.perf_counter()
                tts = make_tts(cfg)
                tts.load(state["voices"])
                logger.info(f"TTS {tts.name} loaded in {time.perf_counter() - t0:.1f}s | {tts.device_info()}")
                _warm_tts(tts, state["voices"])
                state["tts"] = tts
        if mode in ("chat", "assistant") and cfg.tools.enabled:
            _setup_tools(cfg, state)
            hub = state.get("mcp")
            if hub is not None and not await asyncio.to_thread(hub.wait, cfg.tools.mcp_wait_s):
                logger.warning(f"tools: MCP servers not ready after {cfg.tools.mcp_wait_s:.0f}s; "
                               "their tools are added for new connections once they are")
            if llm_factory is None:      # real Ollama: cache the exact system prompt + tool schemas
                await _warm_llm_with_tools(cfg, state)
        logger.info(f"ready: mode={mode}")
        yield
        state["stop_tracker"] = True
        if state.get("mcp") is not None:
            state["mcp"].close()         # stop the MCP server processes

    app = FastAPI(lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/shots", StaticFiles(directory=shots), name="shots")   # screenshots shown in the UI

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/config")
    async def client_config():
        return {"mode": mode, "stt_language": cfg.stt.language, "in_rate": cfg.audio.input_sample_rate,
                "out_rate": cfg.audio.output_sample_rate, "frame_ms": cfg.audio.frame_ms}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )

        from voice_agent.server.serializer import RawPCMSerializer

        await websocket.accept()
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=RawPCMSerializer(),
            ),
        )
        llm = state["llm_factory"]() if state["llm_factory"] else None
        task = build_task(mode, transport, cfg, state["stt"], llm, state["tts"], state["voices"], record_dir,
                          live_provider, state["desktop"], state["vision"])

        @transport.event_handler("on_client_disconnected")
        async def _gone(_t, _ws):
            await task.cancel()

        try:
            await PipelineRunner(handle_sigint=False).run(task)
        finally:
            if llm is not None:
                await llm.aclose()

    return app


async def _warm_llm_with_tools(cfg: Config, state: dict) -> None:
    from datetime import datetime

    from voice_agent.knowledge import build_messages
    from voice_agent.language import directive
    from voice_agent.server.pipelines import make_agent

    agent = make_agent(cfg, state["desktop"], state["vision"])
    if agent is None:
        return
    llm = state["llm_factory"]()
    try:
        msgs = build_messages(f"{cfg.llm.system_prompt}\n\n{agent.system_note()}", [], "hi", datetime.now(), "",
                              directive("en"), cfg.llm.cache_friendly)
        ms = await llm.warmup(msgs, agent.schemas())
        logger.info(f"LLM prompt cache warmed with {len(agent.schemas())} tool schemas in {ms:.0f} ms")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM tool warmup failed: {e}")
    finally:
        await llm.aclose()


def _setup_tools(cfg: Config, state: dict) -> None:
    """Desktop backend (shared), app index in the background, focus tracker, vision."""
    import threading

    from voice_agent.tools.desktop import DryRunDesktop, make_desktop

    if state["desktop"] is None:
        state["desktop"] = make_desktop(cfg.tools.desktop, screenshots_dir=cfg.tools.screenshots_dir,
                                        browser=cfg.tools.browser)
    desk = state["desktop"]
    if state["vision"] is None and cfg.tools.vision and not isinstance(desk, DryRunDesktop):
        from voice_agent.tools.vision import OllamaVision

        state["vision"] = OllamaVision(cfg.llm.host, cfg.tools.vision_model or cfg.llm.model,
                                       keep_alive=cfg.llm.keep_alive, num_ctx=cfg.llm.num_ctx)

    def index():
        t0 = time.perf_counter()
        try:
            n = len(desk.list_apps())
            logger.info(f"tools: desktop={desk.name}, {n} installed apps indexed in "
                        f"{time.perf_counter() - t0:.1f}s, screenshots -> {desk.screenshots_dir}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tools: app index failed ({e}); built-in app names still work")

    def track():   # remember the user's last real window (keyboard target)
        while not state.get("stop_tracker"):
            try:
                desk.remember_target()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.4)

    def start_files():   # start the Filesystem MCP server now, so the first file request is fast
        from voice_agent.tools.file_access import make_files

        if not cfg.tools.files.get("enabled"):
            return
        fs = make_files(cfg.tools.files)
        if fs is None:
            logger.warning("tools: file tools are off: Node.js (npx) not found. See README.md, File tools")
            return
        try:
            t0 = time.perf_counter()
            n = len(fs.list_tools())     # fs.list_tools creates the folder first
            logger.info(f"tools: file tools ready (MCP {fs.client.server_info.get('name', '?')}, {n} server tools) "
                        f"for {fs.root} in {time.perf_counter() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tools: file server failed to start ({e})")

    threading.Thread(target=index, daemon=True).start()
    if not isinstance(desk, DryRunDesktop):
        threading.Thread(target=track, daemon=True).start()
        threading.Thread(target=start_files, daemon=True).start()
        _start_mcp_servers(cfg, state)


def _start_mcp_servers(cfg: Config, state: dict) -> None:
    """Servers from tools.mcp_servers (tools/mcp_bridge.py), started in parallel in the background."""
    from voice_agent.tools.mcp_bridge import shared_hub

    try:
        hub = shared_hub(cfg.tools.mcp_servers)
    except ValueError as e:
        logger.error(f"tools: MCP servers off, bad config.yaml tools.mcp_servers: {e}")
        return
    if hub is None:
        return

    def report(statuses):
        for st in statuses:
            if st.state == "ready":
                names = ", ".join(f"{st.name}__{t['name']}" for t in st.tools) or "no tools"
                logger.info(f"tools: MCP {st.name} ready in {st.seconds:.1f}s "
                            f"({st.server_info.get('name', '?')}): {names}")
                if st.missing:
                    logger.warning(f"tools: MCP {st.name} has no tool(s) {st.missing} (check tools: in config)")
            elif st.state == "failed":
                logger.warning(f"tools: MCP {st.name} failed to start ({st.error})")

    state["mcp"] = hub
    hub.start_background(report)


def _warm_tts(tts, voices) -> None:
    """First synthesis per language compiles kernels; do it before the user talks."""
    t0 = time.perf_counter()
    for lang, text in (("en", "Ready, how can I help you today?"), ("ar", "جاهز، كيف يمكنني مساعدتك اليوم؟")):
        v = next((v for v in voices.values() if v.language == lang), None)
        if v:
            for _ in tts.synthesize(text, v):
                pass
    logger.info(f"TTS warmup {1000 * (time.perf_counter() - t0):.0f} ms")


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    ap = argparse.ArgumentParser(description="Voice agent server")
    ap.add_argument("--mode", choices=MODES, default="transcribe")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="show Pipecat debug logs")
    ap.add_argument("--record-segments", action="store_true",
                    help="save every STT segment to logs/segments/ (wav + json) for diagnosis")
    args = ap.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO")
    cfg = load_config()
    host, port = args.host or cfg.server.host, args.port or cfg.server.port
    logger.info(f"open http://{host}:{port} in Chrome/Edge (mode={args.mode})")
    record_dir = cfg.paths.logs_dir / "segments" if args.record_segments else None
    if record_dir:
        logger.info(f"recording STT segments to {record_dir}")
    uvicorn.run(
        create_app(args.mode, cfg, record_dir=record_dir),
        host=host,
        port=port,
        log_level="warning",
        ws_ping_interval=None,  
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
