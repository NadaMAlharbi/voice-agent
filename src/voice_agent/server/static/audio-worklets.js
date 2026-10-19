// AudioWorklet processors (run on the audio thread).
//
// mic-capture : mic @ context rate -> 16 kHz PCM16, posted in 20 ms frames (640 bytes)
// pcm-player  : ring buffer of float samples @ context rate with a small jitter buffer; "clear"
//               empties it instantly (barge-in); reports buffered ms so the UI can show it.

class MicCapture extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.outRate = options.processorOptions.outRate || 16000;
    this.ratio = sampleRate / this.outRate;       // e.g. 48000/16000 = 3
    this.frameLen = Math.round(this.outRate * 0.02); // 320 samples = 20 ms
    this.frame = new Int16Array(this.frameLen);
    this.n = 0;
    this.pos = 0;     // fractional read position into the input stream
    this.acc = 0;     // box-filter accumulator (cheap anti-aliasing)
    this.accN = 0;
    this.level = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      this.acc += s; this.accN++;
      this.level = Math.max(this.level * 0.9995, Math.abs(s));
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.acc / this.accN));
        this.acc = 0; this.accN = 0;
        this.frame[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.n === this.frameLen) {
          this.port.postMessage({ pcm: this.frame.buffer, level: this.level }, [this.frame.buffer]);
          this.frame = new Int16Array(this.frameLen);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

class PcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(sampleRate * 30); // 30 s ring buffer
    this.r = 0; this.w = 0; this.size = 0;
    this.tick = 0;
    // Small jitter buffer: start (or restart after the queue ran dry) only once a little audio is
    // queued, or when no more audio comes. Streamed TTS pieces can arrive a bit late; without
    // this, every late piece is a click and a tiny gap (choppy speech).
    this.startN = Math.round(sampleRate * 0.06);    // first start: 60 ms queued
    this.resumeN = Math.round(sampleRate * 0.2);    // after running dry: 200 ms queued
    this.need = this.startN;
    this.waiting = true;
    this.idle = 0;                                  // render quanta since the last audio arrived
    this.fade = 0;                                  // samples left in a short fade-in
    this.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "clear") {
        this.r = this.w = this.size = 0; this.playing = false;
        this.waiting = true; this.need = this.startN; this.report(); return;
      }
      if (m.type === "audio") {
        const s = m.samples;
        this.idle = 0;
        for (let i = 0; i < s.length; i++) {
          if (this.size === this.buf.length) break; // overflow: drop
          this.buf[this.w] = s[i];
          this.w = (this.w + 1) % this.buf.length;
          this.size++;
        }
      }
    };
  }
  report() { this.port.postMessage({ bufferedMs: (this.size / sampleRate) * 1000 }); }
  process(_inputs, outputs) {
    const out = outputs[0][0];
    this.idle++;
    if (this.waiting && this.size > 0 && (this.size >= this.need || this.idle * out.length > sampleRate * 0.08)) {
      this.waiting = false;                         // enough queued, or the stream has paused: play
      this.fade = Math.round(sampleRate * 0.002);
    }
    if (!this.waiting && this.size > 0 && !this.playing) {   // sound starts now: report for latency metrics
      this.playing = true;
      this.port.postMessage({ started: true });
    } else if (this.size === 0 && this.playing) {
      this.playing = false;
    }
    const fadeN = Math.round(sampleRate * 0.002);
    for (let i = 0; i < out.length; i++) {
      if (!this.waiting && this.size > 0) {
        let v = this.buf[this.r];
        if (this.fade > 0) { v *= 1 - this.fade / fadeN; this.fade--; }       // 2 ms fade-in: no click
        else if (this.size < fadeN) v *= this.size / fadeN;                      // fade out as it runs dry
        out[i] = v;
        this.r = (this.r + 1) % this.buf.length;
        this.size--;
        if (this.size === 0) { this.waiting = true; this.need = this.resumeN; }  // ran dry: re-buffer
      } else out[i] = 0;
    }
    if (++this.tick % 20 === 0) this.report();
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
registerProcessor("pcm-player", PcmPlayer);
