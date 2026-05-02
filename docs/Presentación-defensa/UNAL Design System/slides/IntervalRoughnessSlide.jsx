const MathText = ({ math, inline = true, style }) => {
  if (!window.katex) return <span>[KaTeX Loading...]</span>;
  const html = window.katex.renderToString(math, { throwOnError: false, displayMode: !inline });
  return <span style={style} dangerouslySetInnerHTML={{ __html: html }} />;
};

const PianoSVG = ({ step }) => {
  // 8 white keys (C to C)
  const whiteKeys = Array.from({ length: 8 }).map((_, i) => (
    <rect key={`w-${i}`} x={i * 45} y={0} width={45} height={180} fill="white" stroke="#1A1A1A" strokeWidth="2" />
  ));
  
  // Black keys positions relative to white keys (C#, D#, F#, G#, A#)
  const blackPositions = [0, 1, 3, 4, 5];
  const blackKeys = blackPositions.map((pos) => (
    <rect key={`b-${pos}`} x={pos * 45 + 30} y={0} width={30} height={110} fill="#1A1A1A" />
  ));

  return (
    <svg width="360" height="200" viewBox="0 0 360 200" style={{ overflow: 'visible' }}>
      {whiteKeys}
      {blackKeys}
      <line x1="360" y1="0" x2="360" y2="180" stroke="#1A1A1A" strokeWidth="4" />
      
      {/* Interval 1 (Red): C (white 0) to C# (black 0) */}
      <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
        <circle cx="22.5" cy="150" r="12" fill="#C0392B" /> {/* C natural */}
        <circle cx="45" cy="90" r="12" fill="#C0392B" />  {/* C# */}
        <line x1="22.5" y1="150" x2="45" y2="90" stroke="#C0392B" strokeWidth="3" strokeDasharray="6,6" />
        <text x="-10" y="125" fill="#C0392B" fontSize="32" fontWeight="bold" fontFamily="'Raleway',sans-serif">1</text>
      </g>
      
      {/* Interval 11 (Green): C# (black 0) to C (white 7) */}
      <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s' }}>
        <circle cx="45" cy="30" r="12" fill="#27AE60" /> {/* C# (top) */}
        <circle cx="337.5" cy="150" r="12" fill="#27AE60" /> {/* C natural (high) */}
        <line x1="45" y1="30" x2="337.5" y2="150" stroke="#27AE60" strokeWidth="3" strokeDasharray="6,6" />
        <text x="180" y="110" fill="#27AE60" fontSize="32" fontWeight="bold" fontFamily="'Raleway',sans-serif">11</text>
      </g>
    </svg>
  );
};

const PlayButton = ({ onClick, visible }) => (
  <div 
    onClick={onClick}
    style={{
      width: 48, height: 48, borderRadius: '50%', backgroundColor: '#E0E0E0',
      display: 'flex', justifyContent: 'center', alignItems: 'center',
      cursor: 'pointer', boxShadow: '0 2px 5px rgba(0,0,0,0.2)',
      opacity: visible ? 1 : 0, transition: 'opacity 0.5s',
      pointerEvents: visible ? 'auto' : 'none'
    }}>
    <svg width="24" height="24" viewBox="0 0 24 24" fill="#444">
      <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
    </svg>
  </div>
);

const RoughnessCurveSVG = ({ step }) => {
  const W = 1200;
  const H = 600;
  const plot = { left: 100, right: 54, top: 28, bottom: 72 };
  const xMax = 12;
  const yMax = 3.2;
  // Same theoretical Sethares curve used by tools/plot_sethares_sweep.py.
  const params = {
    baseFreq: 500,
    nHarmonics: 6,
    decay: 0.88,
    C1: 5.0,
    C2: -5.0,
    A1: -3.51,
    A2: -5.75,
    Dstar: 0.24,
    S1: 0.0207,
    S2: 18.96,
  };

  const pairRoughness = (f1, f2, a1, a2) => {
    const fmin = Math.min(f1, f2);
    const s = params.Dstar / (params.S1 * fmin + params.S2);
    const df = Math.abs(f2 - f1);
    return a1 * a2 * (
      params.C1 * Math.exp(params.A1 * s * df) +
      params.C2 * Math.exp(params.A2 * s * df)
    );
  };

  const roughnessAt = (semitones) => {
    const f1 = params.baseFreq;
    const f2 = params.baseFreq * Math.pow(2, semitones / 12);
    let total = 0;
    for (let k1 = 1; k1 <= params.nHarmonics; k1 += 1) {
      for (let k2 = 1; k2 <= params.nHarmonics; k2 += 1) {
        total += pairRoughness(
          f1 * k1,
          f2 * k2,
          Math.pow(params.decay, k1 - 1),
          Math.pow(params.decay, k2 - 1)
        );
      }
    }
    return total;
  };

  const xScale = (x) => plot.left + (x / xMax) * (W - plot.left - plot.right);
  const yScale = (y) => H - plot.bottom - (y / yMax) * (H - plot.top - plot.bottom);
  const curvePoints = Array.from({ length: 520 }, (_, i) => {
    const x = (i / 519) * xMax;
    return [xScale(x), yScale(roughnessAt(x))];
  });
  const curvePath = curvePoints.map(([x, y], i) => `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`).join(' ');
  const interval1 = { x: xScale(1), y: yScale(roughnessAt(1)), value: roughnessAt(1) };
  const interval11 = { x: xScale(11), y: yScale(roughnessAt(11)), value: roughnessAt(11) };
  const yTicks = [0, 0.5, 1, 1.5, 2, 2.5, 3];
  
  return (
    <svg width="100%" height="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      {[0,1,2,3,4,5,6,7,8,9,10,11,12].map(i => (
        <g key={`grid-x-${i}`}>
          <line x1={xScale(i)} y1={plot.top} x2={xScale(i)} y2={H - plot.bottom} stroke="#D8D8D8" strokeWidth="1.4" />
          <text x={xScale(i)} y={H - 35} fill="#1A1A1A" fontSize="22" textAnchor="middle" fontFamily="'Raleway',sans-serif">{i}</text>
        </g>
      ))}
      {yTicks.map(v => (
        <g key={`grid-y-${v}`}>
          <line x1={plot.left} y1={yScale(v)} x2={W - plot.right} y2={yScale(v)} stroke="#D8D8D8" strokeWidth="1.2" />
          <text x={plot.left - 18} y={yScale(v) + 7} fill="#1A1A1A" fontSize="20" textAnchor="end" fontFamily="'Raleway',sans-serif">{v.toFixed(1)}</text>
        </g>
      ))}
      <line x1={plot.left} y1={plot.top} x2={plot.left} y2={H - plot.bottom} stroke="#1A1A1A" strokeWidth="2.2" />
      <line x1={plot.left} y1={H - plot.bottom} x2={W - plot.right} y2={H - plot.bottom} stroke="#1A1A1A" strokeWidth="2.2" />
      
      <text x="600" y="592" fill="#1A1A1A" fontSize="24" textAnchor="middle" fontFamily="'Raleway',sans-serif">Intervalo a (semitonos)</text>
      <text x="28" y="275" fill="#1A1A1A" fontSize="24" textAnchor="middle" transform="rotate(-90, 28, 275)" fontFamily="'Raleway',sans-serif">Rugosidad sensorial</text>

      <path d={curvePath} fill="none" stroke="#2E86C1" strokeWidth="3.2" strokeLinejoin="round" strokeLinecap="round" />

      <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
        <line x1={plot.left} y1={interval1.y} x2={W - plot.right} y2={interval1.y} stroke="#C0392B" strokeWidth="2.6" strokeDasharray="10,10" />
        <line x1={interval1.x} y1={plot.top} x2={interval1.x} y2={H - plot.bottom} stroke="#1A1A1A" strokeWidth="2.6" strokeDasharray="10,10" />
        <circle cx={interval1.x} cy={interval1.y} r="13" fill="#E42E12" />
        <text x={interval1.x + 18} y={interval1.y - 18} fill="#C0392B" fontSize="22" fontWeight="700" fontFamily="'Raleway',sans-serif">a = 1</text>
      </g>

      <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s' }}>
        <line x1={plot.left} y1={interval11.y} x2={W - plot.right} y2={interval11.y} stroke="#6AA84F" strokeWidth="2.6" strokeDasharray="10,10" />
        <line x1={interval11.x} y1={plot.top} x2={interval11.x} y2={H - plot.bottom} stroke="#1A1A1A" strokeWidth="2.6" strokeDasharray="10,10" />
        <circle cx={interval11.x} cy={interval11.y} r="13" fill="#6AA84F" />
        <text x={interval11.x - 88} y={interval11.y - 18} fill="#4F8D38" fontSize="22" fontWeight="700" fontFamily="'Raleway',sans-serif">a = 11</text>
      </g>
    </svg>
  );
};

const IntervalRoughnessSlide = ({ pageNum = 15, department = "Facultad de Ciencias · Matemáticas Aplicadas" }) => {
  const step = window.useDeckStep(3, 'slide-interval');

  // Audio simulation handlers
  const playSound = (interval) => {
    console.log(`[Audio Simulation] Playing interval ${interval}`);
    try {
      const actx = new (window.AudioContext || window.webkitAudioContext)();
      const f1 = interval === 1 ? 261.63 : 277.18; // C4 for interval 1, C#4 for interval 11
      const f2 = interval === 1 ? 277.18 : 523.25; // C#4 for interval 1, C5 for interval 11
      
      const osc1 = actx.createOscillator();
      const osc2 = actx.createOscillator();
      const gain = actx.createGain();
      
      osc1.type = 'sawtooth';
      osc2.type = 'sawtooth';
      osc1.frequency.value = f1;
      osc2.frequency.value = f2;
      
      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(actx.destination);
      
      gain.gain.setValueAtTime(0.08, actx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.005, actx.currentTime + 1.5);
      
      osc1.start();
      osc2.start();
      osc1.stop(actx.currentTime + 1.5);
      osc2.stop(actx.currentTime + 1.5);
    } catch (e) {
      console.log('Audio playback not supported or blocked', e);
    }
  };

  return (
    <div style={{ position: 'absolute', inset: 0, backgroundColor: 'white', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      
      {/* Top Accent Line */}
      <div style={{ position: 'absolute', top: 0, right: 0, height: 6, width: '60%', backgroundColor: '#E8A020', zIndex: 10 }}></div>
      
      {/* Title */}
      <div style={{ padding: '60px 80px 10px 80px', zIndex: 10 }}>
        <h1 style={{ 
          fontFamily: "'Playfair Display', serif", 
          fontWeight: 800, 
          fontSize: 64, 
          color: '#1A1A1A', 
          margin: 0 
        }}>
          Rugosidad auditiva
        </h1>
      </div>

      {/* Main Content Area - Full screen graph */}
      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        
        {/* Roughness Chart spans entire area */}
        <div style={{ position: 'absolute', inset: '0 40px 64px 40px' }}>
          <RoughnessCurveSVG step={step} />
        </div>

        {/* Piano Schema & Play Buttons overlay top right */}
        <div style={{ 
          position: 'absolute', right: 60, top: 0, 
          border: '2px solid #1A1A1A', 
          backgroundColor: 'white',
          boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
          display: 'flex',
          alignItems: 'center',
          padding: '20px 20px 20px 80px', // Extra left padding for the button
          zIndex: 20
        }}>
          {/* Close button representation */}
          <div style={{
            position: 'absolute', top: 15, left: 15,
            width: 32, height: 32, borderRadius: '50%', backgroundColor: '#444',
            display: 'flex', justifyContent: 'center', alignItems: 'center',
            color: 'white', fontFamily: 'sans-serif', fontSize: 16, cursor: 'pointer'
          }}>
            ✕
          </div>

          <div style={{ position: 'absolute', left: 20, top: 100 }}>
             <PlayButton visible={step >= 2} onClick={() => playSound(1)} />
          </div>
          
          <PianoSVG step={step} />
          
          <div style={{ marginLeft: 20 }}>
             <PlayButton visible={step >= 3} onClick={() => playSound(11)} />
          </div>
        </div>

        {/* Formula overlay bottom left */}
        <div style={{
          position: 'absolute', left: 100, bottom: 200,
          border: '1px solid #E8610A',
          backgroundColor: 'white',
          padding: '12px 28px',
          opacity: step >= 1 ? 1 : 0,
          transform: step >= 1 ? 'translateY(0)' : 'translateY(20px)',
          transition: 'all 0.6s ease',
          zIndex: 20
        }}>
          <MathText 
            math="R(i,j) = a \cdot \left(5e^{-3.51 \cdot S(f_j - f_i)} - 5e^{-5.75 \cdot S(f_j - f_i)}\right)" 
            style={{ fontSize: 32, color: '#1A1A1A' }}
          />
        </div>

      </div>

      <InstitutionalFooter pageNum={pageNum} department={department} />

    </div>
  );
};

Object.assign(window, { IntervalRoughnessSlide });
