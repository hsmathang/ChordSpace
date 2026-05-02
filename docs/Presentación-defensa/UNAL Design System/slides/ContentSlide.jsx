// ContentSlide.jsx — Standard white content slide with title + body area

const ContentSlide = ({
  pageNum = 3,
  title = "Intro musical",
  department = "Facultad de Ciencias · Matemáticas Aplicadas",
  children,
}) => (
  <div style={{position:'absolute',inset:0}}>
    <SlideChrome pageNum={pageNum} department={department}>
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 20 }}>
        <div style={{
          fontFamily: "'Playfair Display','Georgia',serif",
          fontWeight: 800,
          fontSize: 32,
          color: '#1B7A3E',
          lineHeight: 1.1,
        }}>{title}</div>
        <div style={{ flex: 1 }}>{children}</div>
      </div>
    </SlideChrome>
  </div>
);

// Sample content: note table
const NoteTableDemo = () => (
  <div style={{ display: 'flex', gap: 48, alignItems: 'flex-start', marginTop: 16 }}>
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {[
        ['Nota:', 'Do', '{48}'],
        ['Intervalo:', 'Do-Mi', '{48, 52}'],
        ['Acorde:', 'Cm', '{48, 52, 55}'],
      ].map(([a, b, c]) => (
        <div key={a} style={{ display: 'flex', gap: 24, fontFamily: "'Raleway',sans-serif", fontSize: 18, color: '#1A1A1A' }}>
          <span style={{ minWidth: 90, color: '#444' }}>{a}</span>
          <span style={{ minWidth: 60 }}>{b}</span>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 16 }}>{c}</span>
        </div>
      ))}
    </div>
    {/* Circle of fifths mini */}
    <svg width="130" height="130" viewBox="0 0 130 130">
      {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'].map((note, i) => {
        const angle = (i * 30 - 90) * Math.PI / 180;
        const r = 52; const cx = 65; const cy = 65;
        const x = cx + r * Math.cos(angle);
        const y = cy + r * Math.sin(angle);
        return (
          <g key={note}>
            <text x={x} y={y+4} textAnchor="middle" fontSize="9"
              fontFamily="'Raleway',sans-serif" fill={i===0?'#C0392B':i===4?'#1E6BB8':i===7?'#27AE60':'#1A1A1A'}
              fontWeight={[0,4,7].includes(i)?'700':'400'}>{note}</text>
          </g>
        );
      })}
      <circle cx="65" cy="65" r="24" fill="#1A1A1A"/>
      {[0,4,7].map(i => {
        const angle = (i * 30 - 90) * Math.PI / 180;
        const r = 52; const cx = 65; const cy = 65;
        return <circle key={i} cx={cx + r*Math.cos(angle)} cy={cy + r*Math.sin(angle)} r="4"
          fill={i===0?'#C0392B':i===4?'#1E6BB8':'#27AE60'}/>;
      })}
      {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
        const a1 = (i*30-90)*Math.PI/180; const a2 = ((i+1)*30-90)*Math.PI/180;
        const r1=30, r2=42, cx=65, cy=65;
        return <path key={i} d={`M${cx+r1*Math.cos(a1)},${cy+r1*Math.sin(a1)} L${cx+r2*Math.cos(a1)},${cy+r2*Math.sin(a1)}`}
          stroke="#ccc" strokeWidth="0.5"/>;
      })}
      <circle cx="65" cy="65" r="30" fill="none" stroke="#ccc" strokeWidth="0.5"/>
      <circle cx="65" cy="65" r="42" fill="none" stroke="#ccc" strokeWidth="0.5"/>
      {/* numbers */}
      {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
        const angle = (i*30-90)*Math.PI/180; const r=36; const cx=65; const cy=65;
        return <text key={i} x={cx+r*Math.cos(angle)} y={cy+r*Math.sin(angle)+3}
          textAnchor="middle" fontSize="7" fill="white" fontFamily="'Raleway',sans-serif">{i}</text>;
      })}
    </svg>
  </div>
);

Object.assign(window, { ContentSlide, NoteTableDemo });
