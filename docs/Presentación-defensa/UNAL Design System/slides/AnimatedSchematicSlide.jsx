const pianoDotColors = {
  red: '#C0392B',
  blue: '#2980B9',
  green: '#27AE60',
};

const FullPiano = () => {
  const whiteKeys = [
    { midi: 36, note: 'Do' }, { midi: 38, note: 'Re' }, { midi: 40, note: 'Mi' },
    { midi: 41, note: 'Fa' }, { midi: 43, note: 'Sol' }, { midi: 45, note: 'La' },
    { midi: 47, note: 'Si' }, { midi: 48, note: 'Do' }, { midi: 50, note: 'Re' },
    { midi: 52, note: 'Mi' }
  ];
  const blackKeys = [
    { midi: 37, left: 1 }, { midi: 39, left: 2 }, 
    { midi: 42, left: 4 }, { midi: 44, left: 5 }, { midi: 46, left: 6 },
    { midi: 49, left: 8 }, { midi: 51, left: 9 }
  ];

  const pianoWidth = 320;
  const pianoHeight = 120;
  const keyX = (midi) => {
    const index = whiteKeys.findIndex(k => k.midi === midi);
    return ((index + 0.5) / whiteKeys.length) * pianoWidth;
  };

  const voiceRows = [
    { color: 'green', y: 48, keys: [43, 48, 52] },
    { color: 'blue', y: 66, keys: [40, 43, 48] },
    { color: 'red', y: 86, keys: [36, 40, 43] },
  ];

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', border: '3px solid #1A1A1A', display: 'flex', overflow: 'hidden', backgroundColor: 'white' }}>
      {whiteKeys.map((k, i) => (
        <div key={k.midi} style={{ flex: 1, borderRight: i < 9 ? '2px solid #1A1A1A' : 'none', position: 'relative', backgroundColor: 'white' }}>
          <span style={{ position: 'absolute', bottom: 2, width: '100%', textAlign: 'center', fontSize: 13, color: '#1A1A1A', fontWeight: 'bold' }}>{k.note}</span>
          <span style={{ position: 'absolute', top: 52, width: '100%', textAlign: 'center', fontSize: 12, color: '#1A1A1A' }}>{k.midi}</span>
        </div>
      ))}
      <svg viewBox={`0 0 ${pianoWidth} ${pianoHeight}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', zIndex: 3, pointerEvents: 'none' }}>
        {voiceRows.map(row => (
          <g key={row.color}>
            <line
              x1={keyX(row.keys[0])}
              y1={row.y}
              x2={keyX(row.keys[row.keys.length - 1])}
              y2={row.y}
              stroke={pianoDotColors[row.color]}
              strokeWidth="1.6"
            />
            {row.keys.map(midi => (
              <circle
                key={`${row.color}-${midi}`}
                cx={keyX(midi)}
                cy={row.y}
                r="7.2"
                fill={pianoDotColors[row.color]}
                stroke="rgba(255,255,255,0.92)"
                strokeWidth="1.4"
              />
            ))}
          </g>
        ))}
      </svg>
      {blackKeys.map((k) => (
        <div key={k.midi} style={{
          position: 'absolute', left: `${(k.left / 10) * 100}%`, top: 0, transform: 'translateX(-50%)',
          width: '6%', height: '60%', backgroundColor: '#1A1A1A', color: 'white',
          display: 'flex', justifyContent: 'center', alignItems: 'flex-end', paddingBottom: 4, fontSize: 12, zIndex: 2
        }}>
          {k.midi}
        </div>
      ))}
    </div>
  );
};

const AnimatedSchematicSlide = ({ pageNum = 16, department = "Facultad de Ciencias · Matemáticas Aplicadas" }) => {
  const step = window.useDeckStep(6, 'slide-closing');

  return (
    <div style={{ position: 'absolute', inset: 0, backgroundColor: 'white' }}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ 
          position: 'absolute', left: '50%', top: '50%', 
          width: 1600, height: 900, transform: 'translate(-50%, -50%)',
          fontFamily: "'Raleway',sans-serif"
        }}>
          {/* Title */}
          <div style={{ position: 'absolute', top: 10, width: '100%', textAlign: 'center', fontSize: 50, fontWeight: 800, color: '#1A1A1A' }}>
            Esquemático
          </div>

          {/* Grids */}
          <div style={{ position: 'absolute', left: 400, top: 100, bottom: 100, borderLeft: '3px dashed #A0A0A0' }} />
          <div style={{ position: 'absolute', left: 800, top: 100, bottom: 100, borderLeft: '3px dashed #A0A0A0' }} />
          <div style={{ position: 'absolute', left: 1200, top: 100, bottom: 100, borderLeft: '3px dashed #A0A0A0' }} />
          <div style={{ position: 'absolute', left: 50, right: 50, top: 450, borderTop: '3px dashed #A0A0A0' }} />
          <div style={{ position: 'absolute', left: 50, right: 50, bottom: 100, borderTop: '3px dashed #A0A0A0' }} />

          {/* Labels */}
          <div style={{ position: 'absolute', bottom: 40, left: 200, transform: 'translateX(-50%)', fontSize: 32, color: '#666' }}>Codificación</div>
          <div style={{ position: 'absolute', bottom: 40, left: 600, transform: 'translateX(-50%)', fontSize: 32, color: '#666' }}>Diversidad</div>
          <div style={{ position: 'absolute', bottom: 40, left: 1000, transform: 'translateX(-50%)', fontSize: 32, color: '#666' }}>Percepción</div>
          <div style={{ position: 'absolute', bottom: 40, left: 1400, transform: 'translateX(-50%)', fontSize: 32, color: '#666' }}>Visualización</div>

          {/* STEP 1: PIANO */}
          <div style={{ 
            position: 'absolute', left: 40, top: 380, width: 320, height: 120, 
            opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.5s', zIndex: 10 
          }}>
            <FullPiano />
          </div>

          {/* SVG Canvas */}
          <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 5 }}>
            <defs>
              <marker id="arrowGreySchem" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#A0A0A0" />
              </marker>
              <marker id="arrowRedSchem" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#C0392B" />
              </marker>
              <marker id="arrowBlueSchem" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#2980B9" />
              </marker>
              <marker id="arrowGreenSchem" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#27AE60" />
              </marker>
              <marker id="arrowBlackSchem" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A1A1A" />
              </marker>
            </defs>

            {/* CLASSIC PATH (Step 2 & 3) */}
            <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              <path d="M 200,380 C 200,240 300,240 435,240" fill="none" stroke="#1A1A1A" strokeWidth="3" markerEnd="url(#arrowBlackSchem)" />
              <path d="M 450,210 L 440,210 L 440,270 L 450,270" fill="none" stroke="#1A1A1A" strokeWidth="3" />
              <path d="M 440,210 L 490,210" fill="none" stroke="#C0392B" strokeWidth="4" markerEnd="url(#arrowRedSchem)" />
              <path d="M 440,240 L 490,240" fill="none" stroke="#2980B9" strokeWidth="4" markerEnd="url(#arrowBlueSchem)" />
              <path d="M 440,270 L 490,270" fill="none" stroke="#27AE60" strokeWidth="4" markerEnd="url(#arrowGreenSchem)" />
            </g>
            <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              <path d="M 720,240 L 1355,240" fill="none" stroke="#A0A0A0" strokeWidth="4" markerEnd="url(#arrowGreySchem)" />
            </g>

            {/* PROPOSED PATH (Steps 4, 5, 6) */}
            <g style={{ opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              <path d="M 200,500 C 200,640 300,640 435,640" fill="none" stroke="#1A1A1A" strokeWidth="3" markerEnd="url(#arrowBlackSchem)" />
              <path d="M 450,560 L 440,560 L 440,720 L 450,720" fill="none" stroke="#1A1A1A" strokeWidth="3" />
              <path d="M 440,560 L 490,560" fill="none" stroke="#C0392B" strokeWidth="4" markerEnd="url(#arrowRedSchem)" />
              <path d="M 440,640 L 490,640" fill="none" stroke="#2980B9" strokeWidth="4" markerEnd="url(#arrowBlueSchem)" />
              <path d="M 440,720 L 490,720" fill="none" stroke="#27AE60" strokeWidth="4" markerEnd="url(#arrowGreenSchem)" />
            </g>
            <g style={{ opacity: step >= 5 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              <path d="M 720,560 C 780,560 800,580 840,580" fill="none" stroke="#C0392B" strokeWidth="4" markerEnd="url(#arrowRedSchem)" />
              <path d="M 720,640 L 840,640" fill="none" stroke="#2980B9" strokeWidth="4" markerEnd="url(#arrowBlueSchem)" />
              <path d="M 720,720 C 780,720 800,700 840,700" fill="none" stroke="#27AE60" strokeWidth="4" markerEnd="url(#arrowGreenSchem)" />
            </g>
            <g style={{ opacity: step >= 6 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              <path d="M 1150,550 C 1220,550 1250,587 1360,587" fill="none" stroke="#C0392B" strokeWidth="4" markerEnd="url(#arrowRedSchem)" />
              <path d="M 1150,640 C 1220,640 1250,640 1322,640" fill="none" stroke="#2980B9" strokeWidth="4" markerEnd="url(#arrowBlueSchem)" />
              <path d="M 1150,730 C 1220,730 1250,677 1397,677" fill="none" stroke="#27AE60" strokeWidth="4" markerEnd="url(#arrowGreenSchem)" />
            </g>
          </svg>

          {/* CLASSICAL VECTOR */}
          <div style={{ position: 'absolute', left: 240, top: 180, opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s', textAlign: 'center', color: '#1A1A1A', fontSize: 24 }}>
            Vector de<br/>Intervalos<br/>Clásico
          </div>
          <div style={{ position: 'absolute', left: 520, top: 195, opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s', fontSize: 75, fontWeight: 800, color: '#A0A0A0' }}>
            (3,4,5)
          </div>

          {/* PROPOSED VECTORS */}
          <div style={{ position: 'absolute', left: 240, top: 580, opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.5s', textAlign: 'center', color: '#1A1A1A', fontSize: 24 }}>
            Vector de<br/>Intervalos<br/>Propuesto
          </div>
          <div style={{ position: 'absolute', left: 520, top: 515, opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.5s', fontSize: 75, fontWeight: 800, color: '#C0392B' }}>
            (4,3,5)
          </div>
          <div style={{ position: 'absolute', left: 520, top: 595, opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.5s', fontSize: 75, fontWeight: 800, color: '#2980B9' }}>
            (3,5,4)
          </div>
          <div style={{ position: 'absolute', left: 520, top: 675, opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.5s', fontSize: 75, fontWeight: 800, color: '#27AE60' }}>
            (5,4,3)
          </div>

          {/* ROUGHNESS GRAPH */}
          <div style={{ 
            position: 'absolute', left: 850, top: 525, width: 300, height: 230, 
            opacity: step >= 5 ? 1 : 0, transition: 'opacity 0.5s',
            border: '2px solid #ccc', backgroundColor: '#fff', overflow: 'hidden'
          }}>
             {[1,2,3,4,5].map(i => <div key={i} style={{position:'absolute', top: i*38, left:0, right:0, borderTop:'1px solid #eee'}} />)}
             {[1,2,3,4,5,6].map(i => <div key={i} style={{position:'absolute', left: i*42, top:0, bottom:0, borderLeft:'1px solid #eee'}} />)}
             <svg width="300" height="230" style={{ position: 'absolute', inset: 0 }}>
               <path d="M 0,200 Q 20,50 40,80 T 80,120 T 120,130 T 160,110 T 200,160 T 240,80 T 280,140 T 300,100" fill="none" stroke="#85C1E9" strokeWidth="4" />
             </svg>
             <div style={{ position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%, -50%)', fontSize: 45, fontWeight: 800, color: '#A0A0A0', opacity: 0.8 }}>
               Rugosidad
             </div>
          </div>

          {/* TOP GRID (Classical) */}
          <div style={{ 
            position: 'absolute', left: 1300, top: 165, width: 150, height: 150, 
            opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s',
            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gridTemplateRows: '1fr 1fr 1fr', gap: 2, backgroundColor: 'white', border: '2px solid #ccc'
          }}>
            {[...Array(9)].map((_, i) => <div key={i} style={{ backgroundColor: '#E0E0E0' }} />)}
            <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
               <div style={{ fontSize: 16, fontWeight: 'bold', color: '#1A1A1A' }}>C (E/ G)</div>
               <div style={{ width: 14, height: 14, backgroundColor: '#1A1A1A', borderRadius: '50%', marginTop: 4 }} />
            </div>
          </div>

          {/* BOTTOM GRID (Proposed) */}
          <div style={{ 
            position: 'absolute', left: 1300, top: 565, width: 150, height: 150, 
            opacity: step >= 6 ? 1 : 0, transition: 'opacity 0.5s',
            display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gridTemplateRows: '1fr 1fr 1fr', gap: 2, backgroundColor: 'white', border: '2px solid #ccc'
          }}>
            {[...Array(9)].map((_, i) => <div key={i} style={{ backgroundColor: '#E0E0E0' }} />)}
            <div style={{ position: 'absolute', top: 10, left: 10, right: 10, bottom: 10, border: '4px dashed #666', borderRadius: '50%' }} />
            
            <div style={{ position: 'absolute', top: '15%', left: '50%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', gap: 6 }}>
               <div style={{ width: 14, height: 14, backgroundColor: '#C0392B', borderRadius: '50%' }} />
               <div style={{ fontSize: 16, fontWeight: 'bold', color: '#1A1A1A' }}>C</div>
            </div>
            
            <div style={{ position: 'absolute', top: '50%', left: '25%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', gap: 6 }}>
               <div style={{ width: 14, height: 14, backgroundColor: '#2980B9', borderRadius: '50%' }} />
               <div style={{ fontSize: 16, fontWeight: 'bold', color: '#1A1A1A' }}>C/E</div>
            </div>

            <div style={{ position: 'absolute', top: '75%', left: '75%', transform: 'translate(-50%, -50%)', display: 'flex', alignItems: 'center', gap: 6 }}>
               <div style={{ width: 14, height: 14, backgroundColor: '#27AE60', borderRadius: '50%' }} />
               <div style={{ fontSize: 16, fontWeight: 'bold', color: '#1A1A1A' }}>C/G</div>
            </div>
          </div>

        </div>
      </SlideChrome>
    </div>
  );
};

Object.assign(window, { AnimatedSchematicSlide });
