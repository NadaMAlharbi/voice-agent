// Browser client: mic -> WebSocket -> server; server audio/events -> speaker/UI.
// No external libraries (works offline).
import { TurnGate } from "/static/turn-gate.js";
import { Radar, levelOf, stateInfo } from "/static/radar.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
let ctx, micCtx, ws, micNode, playerNode, stream, pingTimer, cfg;
let gate = new TurnGate();
let assistantEl = null;       // chat bubble receiving streamed LLM text
let turnEndAt = null;         // when the browser learned the user's turn ended
let pendingSoundTurn = null;  // accepted turn whose first sound we are waiting for
let micAnalyser = null, outAnalyser = null;   // live audio levels for the radar core
const radar = new Radar($("app"), $("coreLevel"));
if (params.get("theme") === "white") $("app").classList.add("light");

// Small saved settings (language, captions). Storage may be blocked: then defaults are used.
function load(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } }
function save(key, value) { try { localStorage.setItem(key, value); } catch { /* not saved */ } }
let language = load("va.language", "auto");
const languageSaved = load("va.language", null) !== null;
let captionsOn = load("va.captions", "on") !== "off";

// This function adds a line to the activity log (tools, metrics, warnings).
function log(text, cls = "") {
  const div = document.createElement("div");
  div.className = "ev " + cls;
  div.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  $("log").prepend(div);
  while ($("log").childElementCount > 200) $("log").lastChild.remove();
  return div;
}

// This function adds a chat bubble (you or the assistant) and returns it.
function chat(text, who) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  $("chat").append(div);
  $("chat").scrollTop = $("chat").scrollHeight;
  return div;
}

// ---------------------------------------------------------------- captions (big text beside the radar)

let capRest = "";             // streamed text that is not a whole word yet

// Language of a text, like the server's rule: Arabic if 30% of the letters are Arabic,
// or if the first word is Arabic ("افتح Google Chrome").
function langOf(text) {
  const letters = (text.match(/\p{L}/gu) || []);
  const ar = letters.filter((c) => /[\u0600-\u06FF\u0750-\u077F]/.test(c)).length;
  if (!ar) return "en";
  const first = (text.trim().split(/\s+/).find((w) => /\p{L}/u.test(w)) || "");
  return ar / letters.length >= 0.3 || /[\u0600-\u06FF]/.test(first) ? "ar" : "en";
}

// Arabic captions run right to left and sit on the right; English runs left to right.
function captionDir(lang) {
  const dir = lang === "ar" ? "rtl" : "ltr";
  $("captions").dir = dir;
  $("caption").dir = dir;
  $("caption").lang = lang === "ar" ? "ar" : "en";
}

function captionReset(who, lang = "en") {
  $("who").textContent = who;
  $("caption").textContent = "";
  $("caption").classList.remove("cut");
  captionDir(lang);
  capRest = "";
}

function addWords(words, delayStep) {
  const p = $("caption");
  words.forEach((w, i) => {
    if (!w) return;
    const span = document.createElement("span");
    span.className = "w";
    span.style.animationDelay = `${(i * delayStep).toFixed(2)}s`;
    span.textContent = w + " ";
    p.append(span);
  });
  // keep the newest words in view
  while (p.childElementCount > 1 && p.scrollHeight > p.clientHeight + 10) p.firstChild.remove();
}

function captionUser(text) {
  captionReset("YOU SAID", langOf(text));
  addWords(text.split(/\s+/), 0.03);
}

function captionDelta(text) {
  capRest += text;
  const parts = capRest.split(/\s+/);
  capRest = parts.pop();          // the last part may still grow
  addWords(parts, 0);
}

function captionFlush() {
  if (capRest) addWords([capRest], 0);
  capRest = "";
}

function showCaptions(on) {
  captionsOn = on;
  $("captions").classList.toggle("off", !on);
  $("capSwitch").setAttribute("aria-checked", on ? "true" : "false");
  save("va.captions", on ? "on" : "off");
}

// ---------------------------------------------------------------- state

const HINTS = { idle: "Press the microphone and speak Arabic or English", listening: "Speak when you are ready",
                "user speaking": "Pause when you are done", thinking: "Processing on this device",
                speaking: "Speak or press Stop to interrupt", asleep: "Sleeping: say \"HAZEM\" to wake me",
                disconnected: "Disconnected. Press the microphone to connect again" };

// This function updates the status label, the radar animation and the mic button.
function setState(s) {
  $("state").textContent = stateInfo(s).label;
  $("state").dataset.s = s;
  radar.setState(s);
  $("hint").textContent = HINTS[s] ?? "";
  const on = !!ws && ws.readyState === WebSocket.OPEN;
  $("mic").classList.toggle("on", on);
  const label = on ? "Stop listening" : "Start listening";
  $("mic").setAttribute("aria-label", label);
  $("orb").setAttribute("aria-label", label);
}

// PCM16 @ srcRate -> Float32 @ dstRate (linear interpolation)
function pcm16ToFloat(buf, srcRate, dstRate) {
  const src = new Int16Array(buf);
  const ratio = srcRate / dstRate;
  const n = Math.floor(src.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio, i0 = Math.floor(x), f = x - i0;
    const a = src[i0] / 32768, b = (src[i0 + 1] ?? src[i0]) / 32768;
    out[i] = a + (b - a) * f;
  }
  return out;
}

async function start() {
  cfg = await (await fetch("/config")).json();
  $("mode").textContent = cfg.mode;
  if (!languageSaved && cfg.stt_language) showLanguage(cfg.stt_language);   // config default until you pick one
  gate = new TurnGate();
  $("hint").textContent = "Connecting...";

  stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true,
             autoGainControl: params.get("agc") === "1" },
  });
  // Capture runs at 16 kHz: the browser's resampler converts the mic (48/44.1 kHz) cleanly.
  try { micCtx = new AudioContext({ sampleRate: cfg.in_rate, latencyHint: "interactive" }); }
  catch { micCtx = new AudioContext({ latencyHint: "interactive" }); }  // fallback: worklet resamples
  // Playback runs at the TTS rate (24 kHz): the browser resamples to the speakers with a good filter.
  try { ctx = new AudioContext({ sampleRate: cfg.out_rate, latencyHint: "interactive" }); }
  catch { ctx = new AudioContext({ latencyHint: "interactive" }); }
  await micCtx.audioWorklet.addModule("/static/audio-worklets.js");
  await ctx.audioWorklet.addModule("/static/audio-worklets.js");

  micNode = new AudioWorkletNode(micCtx, "mic-capture", { processorOptions: { outRate: cfg.in_rate } });
  playerNode = new AudioWorkletNode(ctx, "pcm-player", { outputChannelCount: [1] });
  const micSrc = micCtx.createMediaStreamSource(stream);
  micSrc.connect(micNode);
  micAnalyser = micCtx.createAnalyser(); micAnalyser.fftSize = 1024;
  micSrc.connect(micAnalyser);                      // side branch: only for the visuals
  outAnalyser = ctx.createAnalyser(); outAnalyser.fftSize = 1024;
  playerNode.connect(outAnalyser);
  outAnalyser.connect(ctx.destination);

  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    setState("listening");
    log(`connected (${cfg.mode}), mic ${micCtx.sampleRate} Hz, playback ${ctx.sampleRate} Hz`);
    sendVoice();
    sendLanguage();
    pingTimer = setInterval(() => ws.send(JSON.stringify({ type: "ping", t: performance.now() })), 2000);
  };
  ws.onclose = () => { log("disconnected"); stop("disconnected"); };
  ws.onmessage = (e) => (typeof e.data === "string" ? onEvent(JSON.parse(e.data)) : onAudio(e.data));

  micNode.port.onmessage = (e) => {
    if (ws?.readyState === WebSocket.OPEN) ws.send(e.data.pcm);
    $("meter").style.width = Math.min(100, e.data.level * 300) + "%";
  };
  playerNode.port.onmessage = (e) => {
    if (e.data.started) { onSoundStarted(); return; }
    $("buffered").textContent = e.data.bufferedMs.toFixed(0);
  };

  $("cmdText").disabled = false; $("cmdSend").disabled = false;
}

function send(obj) { if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj)); }

function showConfirm(question) {
  $("confirmText").textContent = question;
  $("confirm").classList.toggle("on", !!question);
}

const ARGS = (a) => Object.entries(a || {}).filter(([k]) => k !== "language").map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");

function flush() { playerNode?.port.postMessage({ type: "clear" }); }

// Stop button: the server stops the reply (LLM and TTS), and queued audio is dropped now.
function stopReply() {
  flush();
  send({ type: "stop" });
}

function onAudio(buf) {
  // loopback mode has no turns: play everything
  if (cfg.mode !== "loopback" && !gate.onAudio(buf.byteLength)) {
    $("dropped").textContent = (gate.dropped / 48).toFixed(0);  // bytes -> ms at 24 kHz PCM16
    return;
  }
  const samples = pcm16ToFloat(buf, cfg.out_rate, ctx.sampleRate);
  playerNode.port.postMessage({ type: "audio", samples }, [samples.buffer]);
}

// Voice latency as heard: user turn end -> first sample out of the speakers.
function onSoundStarted() {
  if (pendingSoundTurn === null || turnEndAt === null) return;
  const ms = Math.round(performance.now() - turnEndAt);
  log(`turn ${pendingSoundTurn}: first sound ${ms} ms after your turn ended`, "metric");
  if (ws?.readyState === WebSocket.OPEN)
    ws.send(JSON.stringify({ type: "client_metric", turn_id: pendingSoundTurn, first_sound_ms: ms }));
  pendingSoundTurn = null;
}

function onEvent(ev) {
  if (gate.onEvent(ev) === "flush") flush();
  switch (ev.type) {
    case "pong": $("rtt").textContent = (performance.now() - ev.t).toFixed(1); break;
    case "interrupt": break;
    case "cancel_turn":
      log(`turn ${ev.turn_id} cancelled (${ev.reason}); heard: "${ev.heard}"`, "warn");
      if (assistantEl) assistantEl.classList.add("cut");
      $("caption").classList.add("cut");
      if (ev.reason === "stop") setState("listening");
      break;
    case "user_started_speaking": setState("user speaking"); break;
    case "user_stopped_speaking": setState("thinking"); break;
    case "transcript": {
      const lid = ev.lid_top ? `, LID top=${ev.lid_top} ar=${ev.p_ar} en=${ev.p_en}` : "";
      const echo = ev.during_bot_speech ? "  [while assistant was speaking: possible echo]" : "";
      const level = ev.level_lufs != null ? `, voice ${ev.level_lufs} LUFS` : "";
      log(`[${ev.language}] ${ev.text}   (STT ${ev.stt_ms} ms, speech ${ev.speech_s}s${lid}${level})${echo}`, "stt");
      break;
    }
    case "stt_skip": {
      const rej = (ev.rejected || []).map((r) => `"${r.text}" (${r.reason})`).join(", ");
      log(`STT ignored ${ev.audio_s}s segment: ${ev.reason}${rej ? " -> " + rej : ""}`, "metric");
      break;
    }
    case "barge_in_ignored": {
      const extra = ev.echo_score != null ? ` (echo score ${ev.echo_score})` : ev.volume != null ? ` (volume ${ev.volume}, see barge_in.min_volume)` : ev.text ? `: "${ev.text}"` : "";
      log(`not an interruption: ${ev.reason}${ev.why ? " / " + ev.why : ""}${extra}`, "metric");
      break;
    }
    case "lang_guard": log(`language guard removed ${ev.removed} foreign-script chars`, "warn"); break;
    case "user_turn":
      turnEndAt = performance.now();
      if ($("confirm").classList.contains("on") && ev.source !== "button") showConfirm("");
      chat(ev.text, "you");
      captionUser(ev.text);
      if (cfg.mode === "transcribe") setState("listening");
      break;
    case "assistant_start": {
      const kind = ev.intent && ev.intent !== "general"
        ? `, ${ev.intent}${ev.live_data === false ? ": no live data (offline)" : ev.live_data ? ": live data" : ""}` : "";
      if (kind) log(`reply [${ev.language}${kind}]`, "metric");
      assistantEl = chat("", "bot");
      captionReset("ASSISTANT", ev.language === "ar" || ev.language === "en" ? ev.language : "en");
      break;
    }
    case "llm_delta":
      if (assistantEl) assistantEl.textContent += ev.text;
      captionDelta(ev.text);
      $("chat").scrollTop = $("chat").scrollHeight;
      break;
    case "audio_start": setState("speaking"); if (gate.accept === ev.turn_id) pendingSoundTurn = ev.turn_id; break;
    case "tts_chunk": break;
    case "assistant_done":
      captionFlush();
      if ($("state").dataset.s !== "asleep") setState("listening");
      if (ev.speech_end_to_audio_ms != null) {
        $("lat").textContent = ev.speech_end_to_audio_ms;
        log(`turn ${ev.turn_id}: ${ev.speech_end_to_audio_ms} ms from the end of your speech to the reply` +
            `${ev.prefetched ? " (LLM started early)" : ""}${ev.llm_prompt ? `, LLM prompt ${ev.llm_prompt.tokens} new tokens` : ""}`, "metric");
      }
      log(`turn ${ev.turn_id} server: LLM first token ${ev.llm_ttft_ms} ms, first chunk ${ev.first_chunk_ms} ms ` +
          `(${ev.first_chunk_words ?? "-"} words), TTS ${ev.first_synth_ms ?? "-"} ms -> first audio ${ev.first_audio_ms} ms; ` +
          `gaps ${ev.gap_ms} ms, loop lag ${ev.loop_lag_ms} ms, ${ev.chunks} chunks`, "metric");
      if (ev.tts && ev.tts.length) {
        // Speech speed check: above 1.0x the voice is made slower than it plays, so it stutters.
        const rtf = Math.max(...ev.tts.map((c) => (c.synth_ms / 1000) / Math.max(c.audio_s, 0.01)));
        log(`turn ${ev.turn_id} TTS: slowest chunk took ${rtf.toFixed(2)}x its playing time` +
            `${rtf > 0.9 ? " (too slow: speech will stutter)" : ""}`, rtf > 0.9 ? "warn" : "metric");
      }
      break;
    case "voice":
      $("gender").value = ev.gender;
      showGender(ev.gender);
      if (ev.voices) $("gender").title = ev.voices.map((v) => `${v.name}: ${v.engine} ${v.voice}`).join("\n");
      break;
    case "language": showLanguage(ev.language); break;
    case "tool_start": log(`> ${ev.tool}(${ARGS(ev.args)})  [${ev.source}]`, "tool"); break;
    case "tool_result": {
      const what = ev.ok ? JSON.stringify(ev.data) : ev.error;
      const div = log(`${ev.ok ? "ok" : "failed"}: ${ev.tool} ${ev.ms} ms  ${what.slice(0, 200)}`, ev.ok ? "tool" : "tool fail");
      if (ev.image) { const img = document.createElement("img"); img.src = ev.image + "?t=" + Date.now(); div.append(img); }
      break;
    }
    case "confirm_request": showConfirm(ev.question); break;
    case "confirm_resolved": showConfirm(""); log(`${ev.tool}: ${ev.answer ? "confirmed" : "cancelled"}`, "tool"); break;
    case "mode": setState(ev.asleep ? "asleep" : "listening"); log(ev.asleep ? "sleeping: say 'HAZEM' to wake me" : "awake", "metric"); break;
    case "asleep_ignored": log(`(asleep, ignored: "${ev.text}")`, "metric"); break;
    case "timer_done": log(`timer ${ev.label || ""} done`, "turn"); chat(`Reminder: ${ev.label || "timer"}`, "bot"); break;
    case "error": log("error: " + ev.message, "warn"); setState("listening"); break;
    default: log(JSON.stringify(ev));
  }
}

// ---------------------------------------------------------------- settings

function showGender(g) {
  $("vFemale").setAttribute("aria-pressed", g === "female" ? "true" : "false");
  $("vMale").setAttribute("aria-pressed", g === "male" ? "true" : "false");
}

function sendVoice() {
  showGender($("gender").value);
  send({ type: "set_voice", gender: $("gender").value });
}

function showLanguage(lang) {
  language = lang;
  document.querySelectorAll("[data-lang]").forEach((b) =>
    b.setAttribute("aria-pressed", b.dataset.lang === lang ? "true" : "false"));
}

function sendLanguage(remember = false) {
  showLanguage(language);
  if (remember) save("va.language", language);
  send({ type: "set_language", language });
}

function openPanel(which) {
  const pairs = { settings: "setBtn", activity: "actBtn" };
  for (const [panel, btn] of Object.entries(pairs)) {
    const open = panel === which && $(panel).hidden;
    $(panel).hidden = !open;
    $(btn).setAttribute("aria-expanded", open ? "true" : "false");
  }
}

function stop(state = "idle") {
  const w = ws;
  ws = null;
  clearInterval(pingTimer);
  if (w && w.readyState <= 1) { w.onclose = null; w.close(); }
  stream?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  micCtx?.close();
  ctx = micCtx = stream = null;
  $("cmdText").disabled = true; $("cmdSend").disabled = true;
  micAnalyser = outAnalyser = null;
  setState(state);
}

// This function toggles listening (mic button and radar click).
function toggle() {
  if (ws) stop();
  else start().catch((e) => { log("error: " + e.message, "warn"); stop(); $("hint").textContent = "Microphone error: " + e.message; });
}

// Animation loop: live audio levels drive the radar core.
const micBuf = new Float32Array(1024), outBuf = new Float32Array(1024);
function visuals() {
  let mic = 0, out = 0;
  if (micAnalyser) { micAnalyser.getFloatTimeDomainData(micBuf); mic = levelOf(micBuf); }
  if (outAnalyser) { outAnalyser.getFloatTimeDomainData(outBuf); out = levelOf(outBuf); }
  radar.setLevels(mic, out);
  requestAnimationFrame(visuals);
}
requestAnimationFrame(visuals);
setInterval(() => { $("clock").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }, 1000);

setState("idle");
showLanguage(language);
showCaptions(captionsOn);
$("mic").onclick = toggle;
$("orb").onclick = toggle;
$("clear").onclick = stopReply;
$("vFemale").onclick = () => { $("gender").value = "female"; sendVoice(); };
$("vMale").onclick = () => { $("gender").value = "male"; sendVoice(); };
$("gender").onchange = sendVoice;
document.querySelectorAll("[data-lang]").forEach((b) => { b.onclick = () => { language = b.dataset.lang; sendLanguage(true); }; });
$("capSwitch").onclick = () => showCaptions(!captionsOn);
$("setBtn").onclick = () => openPanel("settings");
$("setClose").onclick = () => openPanel("");
$("actBtn").onclick = () => openPanel("activity");
$("actClose").onclick = () => openPanel("");
$("cmd").onsubmit = (e) => {
  e.preventDefault();
  const text = $("cmdText").value.trim();
  if (text) { send({ type: "text", text }); $("cmdText").value = ""; }
};
$("confirmYes").onclick = () => { send({ type: "confirm", answer: true }); showConfirm(""); };
$("confirmNo").onclick = () => { send({ type: "confirm", answer: false }); showConfirm(""); };
