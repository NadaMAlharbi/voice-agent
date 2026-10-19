// Stale-audio protection for the browser. Pure logic, unit-tested with Node.
//
// Audio from the server carries no header; instead the server sends, in order
// with the audio, {"type":"audio_start","turn_id":N} before turn N's first packet.
// Rules:
//   - audio plays only while a turn is accepted
//   - "interrupt" (user started speaking) stops accepting: flush now
//   - "cancel_turn" N marks every turn <= N dead; an audio_start for a dead turn is ignored
// So packets of an old turn that were already in flight when the user barged in are dropped.

export class TurnGate {
  constructor() {
    this.accept = null;      // turn_id currently allowed to play
    this.maxCancelled = 0;
    this.dropped = 0;        // bytes of stale audio dropped
  }

  // Returns "flush" when playback must be cleared immediately, else null.
  onEvent(ev) {
    switch (ev.type) {
      case "interrupt":
        this.accept = null;
        return "flush";
      case "cancel_turn":
        this.maxCancelled = Math.max(this.maxCancelled, ev.turn_id);
        if (this.accept !== null && this.accept <= ev.turn_id) {
          this.accept = null;
          return "flush";
        }
        return null;
      case "audio_start":
        if (ev.turn_id > this.maxCancelled) this.accept = ev.turn_id;
        return null;
      default:
        return null;
    }
  }

  // true = play it, false = stale, drop it
  onAudio(byteLength) {
    if (this.accept === null) {
      this.dropped += byteLength;
      return false;
    }
    return true;
  }
}
