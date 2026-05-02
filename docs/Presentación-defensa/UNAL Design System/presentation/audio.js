'use strict';
// Web Audio engine — UNAL Thesis Defense
// Complex tones (H=6 harmonics, δ=0.88 decay) per Sethares model

window.AudioEngine = (() => {
  let ctx = null;

  const midiToFreq = n => 440 * Math.pow(2, (n - 69) / 12);

  function getCtx() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === 'suspended') ctx.resume();
    return ctx;
  }

  // Play a single complex tone (harmonic series)
  function playComplex(freq, dur, vol, delay) {
    const ac = getCtx();
    const t = ac.currentTime + (delay || 0);
    const H = 6, delta = 0.88;
    const master = ac.createGain();
    master.gain.setValueAtTime(0, t);
    master.gain.linearRampToValueAtTime(vol, t + 0.015);
    master.gain.setValueAtTime(vol, t + Math.max(dur * 0.15, 0.04));
    master.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    master.connect(ac.destination);
    for (let k = 1; k <= H; k++) {
      const o = ac.createOscillator();
      const g = ac.createGain();
      o.type = 'sine';
      o.frequency.value = k * freq;
      g.gain.value = Math.pow(delta, k - 1) / H;
      o.connect(g); g.connect(master);
      o.start(t); o.stop(t + dur + 0.05);
    }
  }

  function playNote(midi, dur = 1.5) {
    playComplex(midiToFreq(midi), dur, 0.22);
  }

  function playChord(midis, dur = 2.0) {
    const vol = Math.min(0.22 / Math.sqrt(midis.length), 0.1);
    midis.forEach(n => playComplex(midiToFreq(n), dur, vol));
  }

  function playInterval(n1, n2, dur = 2.2) {
    playChord([n1, n2], dur);
  }

  // Play all 13 intervals (0..12) from baseNote, sequentially
  let _timers = [];
  function cancelAll() {
    _timers.forEach(clearTimeout);
    _timers = [];
  }

  function playAllIntervals(baseNote = 48, stepMs = 1800, onStep = null) {
    cancelAll();
    for (let i = 0; i <= 12; i++) {
      const t = setTimeout(() => {
        playChord([baseNote, baseNote + i], 1.5);
        if (onStep) onStep(i);
      }, i * stepMs);
      _timers.push(t);
    }
  }

  // Interval names for display
  const INTERVAL_NAMES = [
    'Unísono','2ª menor','2ª mayor','3ª menor','3ª mayor',
    '4ª justa','Tritono','5ª justa','6ª menor','6ª mayor',
    '7ª menor','7ª mayor','Octava'
  ];

  const ROUGHNESS_ORDER = [12,7,5,4,9,3,8,10,2,6,11,1,0]; // approx. consonance order

  return { playNote, playChord, playInterval, playAllIntervals, cancelAll, midiToFreq, INTERVAL_NAMES };
})();
