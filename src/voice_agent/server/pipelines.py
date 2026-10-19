"""Pipeline builders. Each mode is one small, testable pipeline.

  loopback:   mic -> speaker
  transcribe: mic -> STT -> user turn (Silero VAD + Smart Turn) -> browser
  chat:       ... user turn -> Ollama, streamed text replies
  assistant:  ... user turn -> Ollama -> chunker -> TTS -> browser audio,
              with barge-in (see response.py)
"""

from __future__ import annotations

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from voice_agent.config import Config
from voice_agent.server.processors import (
    ClientMessageHandler,
    EchoProcessor,
    LocalSTTService,
    TurnReporter,
)
from voice_agent.server.bargein import BargeInStartStrategy, PlaybackMonitor
from voice_agent.server.response import ResponseProcessor, Session

MODES = ("loopback", "transcribe", "chat", "assistant")


def user_aggregator(cfg: Config, monitor: PlaybackMonitor | None = None,
                    context: LLMContext | None = None) -> LLMContextAggregatorPair:
    """Silero VAD + Smart Turn v3, with the same settings as `vad-test` (config.yaml vad/turn).

    A turn starts on VAD only (see bargein.py). Pipecat's default would also start a
    turn on any late transcript, which caused unrelated replies, so it is not used.

    empty_user_turn=None: Pipecat's "empty turn recovery" would run the LLM again after
    a cough or noise with no transcript, and re-answer the previous question.
    No transcript means no reply.
    """
    params = LLMUserAggregatorParams(
        empty_user_turn=None,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(**vars(cfg.vad))),
        user_turn_strategies=UserTurnStrategies(
            start=[BargeInStartStrategy(monitor, **vars(cfg.barge_in))],
            stop=[TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(**vars(cfg.turn)))
            )]
        ),
    )
    return LLMContextAggregatorPair(context or LLMContext(), user_params=params)


def make_agent(cfg: Config, desktop, vision=None):
    """Per connection: tools bound to the shared desktop backend (timers belong to the session)."""
    from voice_agent.server.agent import Agent
    from voice_agent.tools.file_access import make_files
    from voice_agent.tools.toolbox import Toolbox

    if desktop is None or not cfg.tools.enabled:
        return None
    tb = Toolbox(desktop, cfg.tools.data_dir, vision=vision, allow_power=cfg.tools.allow_power,
                 home_city=cfg.tools.home_city, files=make_files(cfg.tools.files))
    return Agent(tb, use_router=cfg.tools.router, max_steps=cfg.tools.max_steps)


def build_task(mode: str, transport, cfg: Config, stt_engine=None, llm=None, tts=None,
               voices=None, record_dir=None, live_provider=None, desktop=None, vision=None) -> PipelineTask:
    """llm must be a fresh client per connection (it owns an HTTP connection).

    desktop: tools/desktop.py backend shared by all connections (None = no computer control).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    session = Session(gender=cfg.tts.gender, language=cfg.tts.language)
    if mode == "loopback":
        processors = [transport.input(), ClientMessageHandler(), EchoProcessor(), transport.output()]
    else:
        if stt_engine is None:
            raise ValueError(f"{mode} mode needs an STT engine")
        monitor = PlaybackMonitor()   # what the assistant plays, for echo detection
        context = LLMContext()
        agg = user_aggregator(cfg, monitor, context)
        processors = [
            transport.input(),
            ClientMessageHandler(session, voices),
            LocalSTTService(stt_engine, cfg.stt.language, record_dir=record_dir, session=session),
            agg.user(),
            TurnReporter(swallow_context=(mode == "transcribe")),
        ]
        if mode in ("chat", "assistant"):
            if llm is None or (mode == "assistant" and tts is None):
                raise ValueError(f"{mode} mode needs llm{' and tts' if mode == 'assistant' else ''}")
            processors.append(ResponseProcessor(
                llm, tts if mode == "assistant" else None, voices, session, cfg.chunker,
                live_provider=live_provider, monitor=monitor,
                agent=make_agent(cfg, desktop, vision), context=context,
            ))
        processors.append(transport.output())

    return PipelineTask(
        Pipeline(processors),
        params=PipelineParams(
            audio_in_sample_rate=cfg.audio.input_sample_rate,
            audio_out_sample_rate=cfg.audio.output_sample_rate,
            enable_metrics=True,
        ),
        idle_timeout_secs=None,
        enable_rtvi=False,  # we use our own small JSON protocol (serializer.py)
    )
