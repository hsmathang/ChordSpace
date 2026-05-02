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
  // Chart spans 100% width and height using viewBox 0 0 1200 600
  // Origin X: 100, Y: 550
  // X scale: 1 unit = 80px (12 units = 960px). Ends at X=1060
  // Y scale: 1 unit = 140px. 3.5 units = 490px. Y max at Y=60.
  
  // Custom path approximating the peaks and dips (consonances) exactly
  const curvePath = "M 100 550 C 110 100, 140 0, 180 160 C 220 320, 290 350, 340 370 C 350 370, 360 400, 370 380 C 390 350, 410 350, 420 390 C 440 320, 470 320, 500 420 C 530 350, 550 340, 580 340 C 620 340, 640 460, 660 350 C 690 330, 710 330, 740 360 C 780 430, 800 370, 820 370 C 860 370, 890 320, 940 320 C 970 320, 980 200, 980 200 C 1000 130, 1040 450, 1060 450 C 1080 200, 1100 150, 1140 230";
  
  return (
    <svg width="100%" height="100%" viewBox="0 0 1200 600" preserveAspectRatio="none">
      {/* Grid lines vertical */}
      {[0,1,2,3,4,5,6,7,8,9,10,11,12].map(i => (
        <g key={`grid-x-${i}`}>
          <line x1={100 + i*80} y1="0" x2={100 + i*80} y2="550" stroke="#CCCCCC" strokeWidth="2" />
          <text x={100 + i*80} y="580" fill="#1A1A1A" fontSize="18" textAnchor="middle" fontFamily="'Raleway',sans-serif">{i}</text>
        </g>
      ))}
      <line x1="100" y1="0" x2="100" y2="550" stroke="#1A1A1A" strokeWidth="2" />
      <line x1="100" y1="550" x2="1160" y2="550" stroke="#1A1A1A" strokeWidth="2" />
      
      {/* Y-axis labels */}
      {[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0].map(v => (
        <g key={`grid-y-${v}`}>
          <line x1="90" y1={550 - v*140} x2="100" y2={550 - v*140} stroke="#1A1A1A" strokeWidth="2" />
          <text x="80" y={550 - v*140 + 6} fill="#1A1A1A" fontSize="18" textAnchor="end" fontFamily="'Raleway',sans-serif">{v.toFixed(1)}</text>
        </g>
      ))}
      
      {/* Axis Titles */}
      <text x="600" y="595" fill="#1A1A1A" fontSize="20" textAnchor="middle" fontFamily="'Raleway',sans-serif">Interval (semitones)</text>
      <text x="-40" y="275" fill="#1A1A1A" fontSize="20" textAnchor="middle" transform="rotate(-90, -40, 275)" fontFamily="'Raleway',sans-serif">Sensory dissonance</text>

      {/* Main Curve */}
      <path d={curvePath} fill="none" stroke="#1E6BB8" strokeWidth="3" strokeLinejoin="round" />

      {/* Interval 1 (Red) */}
      <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
        {/* Y value for interval 1 is 2.45 = 550 - 2.45*140 = 207 */}
        <line x1="100" y1={207} x2="1200" y2={207} stroke="#C0392B" strokeWidth="2.5" strokeDasharray="10,10" />
        <circle cx={180} cy={207} r="12" fill="#C0392B" />
      </g>

      {/* Interval 11 (Green) */}
      <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s' }}>
        {/* Y value for interval 11 is 1.25 = 550 - 1.25*140 = 375 */}
        <line x1="100" y1={375} x2="1200" y2={375} stroke="#27AE60" strokeWidth="2.5" strokeDasharray="10,10" />
        <circle cx={980} cy={375} r="12" fill="#27AE60" />
      </g>
    </svg>
  );
};

const IntervalRoughnessSlide = ({ pageNum = 15 }) => {
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
        <div style={{ position: 'absolute', inset: '0 40px 20px 40px' }}>
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
          position: 'absolute', left: 100, bottom: 60,
          border: '1px solid #E8610A',
          backgroundColor: 'white',
          padding: '15px 35px',
          opacity: step >= 1 ? 1 : 0,
          transform: step >= 1 ? 'translateY(0)' : 'translateY(20px)',
          transition: 'all 0.6s ease',
          zIndex: 20
        }}>
          <MathText 
            math="R(i,j) = a \cdot \left(5e^{-3.51 \cdot S(f_j - f_i)} - 5e^{-5.75 \cdot S(f_j - f_i)}\right)" 
            style={{ fontSize: 38, color: '#1A1A1A' }}
          />
        </div>

      </div>

      {/* Footer Strip */}
      <div style={{ height: 48, backgroundColor: '#F0EBE0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 40px', position: 'relative', zIndex: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 15 }}>
          <div style={{ display: 'flex', gap: 4 }}>
            <div style={{ width: 12, height: 4, backgroundColor: '#2DC56E' }}></div>
            <div style={{ width: 30, height: 4, backgroundColor: '#2DC56E' }}></div>
          </div>
          <span style={{ fontFamily: "'EB Garamond', serif", fontStyle: 'italic', fontSize: 14, color: '#1B7A3E' }}>
            Facultad de Ciencias · Matemáticas Aplicadas
          </span>
        </div>
        <div style={{ fontFamily: "'Raleway', sans-serif", fontSize: 14, color: '#1A1A1A' }}>
          {pageNum}
        </div>
      </div>

    </div>
  );
};

Object.assign(window, { IntervalRoughnessSlide });
