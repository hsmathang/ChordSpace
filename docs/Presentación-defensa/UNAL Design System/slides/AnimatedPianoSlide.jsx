const CircleOfFifthsSVG = ({ size = 280 }) => (
  <svg width={size} height={size} viewBox="0 0 130 130">
    <circle cx="65" cy="65" r="64" fill="none" stroke="#1A1A1A" strokeWidth="0.8"/>
    <circle cx="65" cy="65" r="42" fill="none" stroke="#1A1A1A" strokeWidth="0.8"/>
    <circle cx="65" cy="65" r="24" fill="#1A1A1A"/>
    
    {/* Outer division lines */}
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const a1 = (i*30-75)*Math.PI/180; 
      const r1=42, r2=64, cx=65, cy=65;
      return <path key={`out-${i}`} d={`M${cx+r1*Math.cos(a1)},${cy+r1*Math.sin(a1)} L${cx+r2*Math.cos(a1)},${cy+r2*Math.sin(a1)}`}
        stroke="#1A1A1A" strokeWidth="0.8"/>;
    })}

    {/* Inner division lines */}
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const a1 = (i*30-75)*Math.PI/180; 
      const r1=24, r2=42, cx=65, cy=65;
      return <path key={`in-${i}`} d={`M${cx+r1*Math.cos(a1)},${cy+r1*Math.sin(a1)} L${cx+r2*Math.cos(a1)},${cy+r2*Math.sin(a1)}`}
        stroke="#1A1A1A" strokeWidth="0.8"/>;
    })}

    {/* Note texts */}
    {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'].map((note, i) => {
      const angle = (i * 30 - 90) * Math.PI / 180;
      const r = 53; const cx = 65; const cy = 65;
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      return (
        <text key={note} x={x} y={y+4} textAnchor="middle" fontSize="11"
          fontFamily="'Raleway',sans-serif" fill="#1A1A1A">{note}</text>
      );
    })}

    {/* Numbers */}
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const angle = (i*30-90)*Math.PI/180; const r=33; const cx=65; const cy=65;
      return <text key={i} x={cx+r*Math.cos(angle)} y={cy+r*Math.sin(angle)+3}
        textAnchor="middle" fontSize="10" fill="white" fontFamily="'Raleway',sans-serif">{i}</text>;
    })}
  </svg>
);

const DataTable = ({ step }) => (
  <div style={{ 
    display: 'flex', flexDirection: 'column', gap: 10
  }}>
    {[
      ['Nota:', 'Do', '{48}'],
      ['Intervalo:', 'Do-Mi', '{48, 52}'],
      ['Acorde:', 'CM', '{48, 52, 55}'],
    ].map(([a, b, c]) => (
      <div key={a} style={{ display: 'flex', gap: 40, fontFamily: "'Raleway',sans-serif", fontSize: 36, color: '#1A1A1A' }}>
        <span style={{ minWidth: 200, color: '#1A1A1A' }}>{a}</span>
        <span style={{ minWidth: 120 }}>{b}</span>
        <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 32 }}>{c}</span>
      </div>
    ))}
  </div>
);

const PianoKeyboard = ({ step }) => {
  const whiteKeys = [
    { midi: 36, note: 'Do' }, { midi: 38, note: 'Re' }, { midi: 40, note: 'Mi' },
    { midi: 41, note: 'Fa' }, { midi: 43, note: 'Sol' }, { midi: 45, note: 'La' }, { midi: 47, note: 'Si' },
    { midi: 48, note: 'Do', red: true, class: 0 }, { midi: 50, note: 'Re' }, { midi: 52, note: 'Mi', red: true, class: 4 },
    { midi: 53, note: 'Fa' }, { midi: 55, note: 'Sol', red: true, class: 7 }, { midi: 57, note: 'La' }, { midi: 59, note: 'Si' },
    { midi: 60, note: 'Do' }
  ];
  
  const blackKeys = [
    { midi: 37, note: 'Do#', offset: 1 }, { midi: 39, note: 'Re#', offset: 2 },
    { midi: 42, note: 'Fa#', offset: 4 }, { midi: 44, note: 'Sol#', offset: 5 }, { midi: 46, note: 'La#', offset: 6 },
    { midi: 49, note: 'Do#', offset: 8 }, { midi: 51, note: 'Re#', offset: 9 },
    { midi: 54, note: 'Fa#', offset: 11 }, { midi: 56, note: 'Sol#', offset: 12 }, { midi: 58, note: 'La#', offset: 13 }
  ];

  return (
    <div style={{ 
      position: 'relative', width: '100%', height: 320, 
      clipPath: step === 0 ? 'inset(0 53.333% 0 0)' : 'inset(0 0 0 0)',
      transition: 'clip-path 1.2s cubic-bezier(0.2, 0.8, 0.2, 1)'
    }}>
      <div style={{ position: 'absolute', top: 40, left: 0, width: '100%', height: 200, display: 'flex', border: '4px solid #1A1A1A', borderRight: 'none' }}>
        
        {/* White Keys */}
        {whiteKeys.map((wk, i) => (
          <div key={wk.midi} style={{ 
            flex: 1, borderRight: '4px solid #1A1A1A', position: 'relative',
            display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 15
          }}>
            <span style={{ 
              fontFamily: "'Raleway',sans-serif", fontSize: 22, color: '#1A1A1A', 
              opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' 
            }}>{wk.midi}</span>

            <span style={{ 
              fontFamily: "'Raleway',sans-serif", fontSize: 26, color: '#1A1A1A', fontWeight: 600,
              opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s', marginTop: 10 
            }}>{wk.note}</span>

            {wk.red && (
              <div style={{ 
                position: 'absolute', top: 130, width: 20, height: 20, borderRadius: 10, backgroundColor: '#ff1100',
                opacity: step >= 3 ? 1 : 0, transform: step >= 3 ? 'scale(1)' : 'scale(0.5)', transition: 'all 0.5s'
              }} />
            )}
          </div>
        ))}

        {/* Black Keys */}
        {blackKeys.map(bk => (
          <div key={bk.midi} style={{ 
            position: 'absolute', left: `calc(${(bk.offset / 15) * 100}% - 2%)`, width: '4%', height: '62%', 
            backgroundColor: '#1A1A1A', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center',
            paddingBottom: 12
          }}>
            <span style={{ 
              position: 'absolute', top: -38, fontFamily: "'Raleway',sans-serif", fontSize: 24, color: '#1A1A1A', fontWeight: 600,
              opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s', whiteSpace: 'nowrap'
            }}>{bk.note}</span>

            <span style={{ 
              fontFamily: "'Raleway',sans-serif", fontSize: 22, color: 'white', fontWeight: 600,
              opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' 
            }}>{bk.midi}</span>
          </div>
        ))}
      </div>

      {/* Red Classes below keyboard */}
      {whiteKeys.map((wk, i) => wk.red && (
        <span key={`red-${wk.midi}`} style={{
          position: 'absolute', top: 250, left: `calc(${(i + 0.5) / 15 * 100}%)`, transform: 'translateX(-50%)',
          fontFamily: "'Raleway',sans-serif", fontSize: 28, color: '#ff1100', fontWeight: 700,
          opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s'
        }}>{wk.class}</span>
      ))}

      {/* Octave dividers */}
      <div style={{ position: 'absolute', top: 40, bottom: 20, left: `calc(${7/15 * 100}%)`, width: 4, backgroundColor: '#1A1A1A' }} />
      <div style={{ position: 'absolute', top: 40, bottom: 20, left: `calc(${14/15 * 100}%)`, width: 4, backgroundColor: '#1A1A1A' }} />
    </div>
  )
}

const AnimatedPianoSlide = ({
  pageNum = 11,
  department = "Facultad de Ciencias · Matemáticas Aplicadas"
}) => {
  const step = window.useDeckStep(5, 'slide-content');

  return (
    <div style={{position:'absolute',inset:0}}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '40px 80px 0' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', width: '100%' }}>
            <div style={{
              fontFamily: "'Playfair Display','Georgia',serif",
              fontWeight: 800,
              fontSize: 110,
              color: '#1A1A1A',
              lineHeight: 1.1,
              marginTop: 40
            }}>
              Intro musical
            </div>
            
            <div style={{ 
              opacity: step >= 5 ? 1 : 0, 
              transform: step >= 5 ? 'scale(1)' : 'scale(0.95)',
              transition: 'all 0.8s ease',
              marginRight: 20 
            }}>
              <CircleOfFifthsSVG size={360} />
            </div>
          </div>

          <div style={{ 
            marginTop: -160, 
            marginLeft: 300,
            opacity: step >= 4 ? 1 : 0, 
            transform: step >= 4 ? 'translateY(0)' : 'translateY(-10px)',
            transition: 'all 0.8s ease' 
          }}>
            <DataTable step={step} />
          </div>

          <div style={{ flex: 1, display: 'flex', alignItems: 'flex-end', paddingBottom: 60, width: '100%' }}>
             <PianoKeyboard step={step} />
          </div>

        </div>
      </SlideChrome>
    </div>
  );
};

Object.assign(window, { AnimatedPianoSlide });
