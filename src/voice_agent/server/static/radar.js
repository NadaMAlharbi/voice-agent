// Radar dish state for the page: a CSS class per assistant state, and the core grows with the live voice.
// The animation itself is CSS (styles.css). No external libraries.

const STATES = {
  idle:            { cls: "s-idle",   label: "READY" },
  listening:       { cls: "s-listen", label: "LISTENING" },
  "user speaking": { cls: "s-hear",   label: "LISTENING" },
  thinking:        { cls: "s-think",  label: "THINKING" },
  speaking:        { cls: "s-speak",  label: "SPEAKING" },
  asleep:          { cls: "s-asleep", label: "ASLEEP" },
  disconnected:    { cls: "s-off",    label: "OFFLINE" },
};
const CLASSES = Object.values(STATES).map((s) => s.cls);

export function stateInfo(s) {
  return STATES[s] || STATES.idle;
}

// Loudness of audio samples, 0..1.
export function levelOf(samples) {
  let s = 0;
  for (let i = 0; i < samples.length; i++) s += samples[i] * samples[i];
  return Math.min(1, Math.sqrt(s / samples.length) * 4);
}

export class Radar {
  constructor(root, core) {
    this.root = root;          // element that gets the state class
    this.core = core;          // element whose --lvl follows the voice
    this.state = "idle";
    this.level = 0;
  }

  setState(s) {
    this.state = STATES[s] ? s : "idle";
    this.root.classList.remove(...CLASSES);
    this.root.classList.add(stateInfo(this.state).cls);
  }

  // mic = user's voice, out = assistant's voice (0..1). The voice that matters now drives the core.
  setLevels(mic, out) {
    const target = this.state === "speaking" ? out : this.state === "user speaking" ? mic : 0;
    this.level += (target - this.level) * (target > this.level ? 0.5 : 0.15);
    this.core.style.setProperty("--lvl", this.level.toFixed(3));
  }
}
