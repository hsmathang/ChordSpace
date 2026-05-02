// SplitSlide.jsx — Orange right panel / white left panel

const SplitSlide = ({
  pageNum = 5,
  department = "Facultad de Ciencias · Matemáticas Aplicadas",
  leftContent,
  rightTitle = "PC-Set Theory",
  rightContent,
}) => (
  <div style={{position:'absolute',inset:0,display:'flex',overflow:'hidden'}}>
      {/* Left white panel */}
      <div style={{ flex: '0 0 40%', background: '#fff', padding: '52px 32px 52px 72px', display: 'flex', flexDirection: 'column', justifyContent: 'center', position: 'relative' }}>
        <div style={{ position: 'absolute', top: 14, left: 18, fontSize: 13, fontFamily: "'Raleway',sans-serif", color: '#777' }}>{pageNum}</div>
        {leftContent}
      </div>
      {/* Right orange panel */}
      <div style={{ flex: '0 0 60%', background: '#E8610A', padding: '40px 48px', display: 'flex', flexDirection: 'column', gap: 28 }}>
        <div style={{
          fontFamily: "'Raleway',sans-serif",
          fontWeight: 700,
          fontSize: 22,
          color: '#fff',
          textTransform: 'uppercase',
          letterSpacing: '0.12em',
          borderBottom: '1px solid rgba(255,255,255,0.3)',
          paddingBottom: 12,
        }}>{rightTitle}</div>
        {rightContent}
      </div>
      {/* Gold accent top-right */}
      <div style={{ position: 'absolute', top: 16, right: 0, width: '28%', height: 2, background: '#E8A020' }} />
      {/* Bottom green lines */}
      <div style={{ position: 'absolute', bottom: 58, left: 28, width: 58, height: 3, background: '#2DC56E' }} />
      <div style={{ position: 'absolute', bottom: 50, left: 28, width: 34, height: 3, background: '#2DC56E' }} />
      {/* Footer */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, height: 48,
        background: '#F0EBE0',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '0 22px',
      }}>
        <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 12, color: '#1B7A3E', fontStyle: 'italic', lineHeight: 1.35, fontWeight: 600 }}>
          {department.split('·').map((d, i) => <div key={i}>{d.trim()}</div>)}
        </div>
        <img
          src="assets/escudo-un-2016.jpg"
          alt="Escudo Universidad Nacional de Colombia"
          style={{ width: 50, height: 50, objectFit: 'contain', mixBlendMode: 'multiply', opacity: 0.82 }}
        />
      </div>
  </div>
);

// Demo content for SplitSlide
const SplitDemoLeft = () => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
    <div style={{ fontFamily: "'Playfair Display',serif", fontWeight: 800, fontSize: 24, color: '#1B7A3E', marginBottom: 8 }}>Chords</div>
    {['C={0,4,7}','Am={9,0,4}','B dim={11,2,5}'].map(c => (
      <div key={c} style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 14, color: '#1A1A1A' }}>{c}</div>
    ))}
  </div>
);

const SplitDemoRight = () => {
  const rows = [
    { label:'iStruct', vals:['(4,7)', '→', '(0,0,1,1,1,0)'], color:'#fff' },
    { label:'iStruct', vals:['(3,7)', '→', '(0,0,2,0,0,1)'], color:'#fff' },
    { label:'iStruct', vals:['(3,6)', '→', ''], color:'#fff' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: 4 }}>
        <span style={{ fontFamily:"'Raleway',sans-serif", fontWeight:700, fontSize:14, color:'rgba(255,255,255,0.7)', textTransform:'uppercase', letterSpacing:'0.08em' }}>iStruct</span>
        <span style={{ fontFamily:"'Raleway',sans-serif", fontWeight:700, fontSize:14, color:'rgba(255,255,255,0.7)', textTransform:'uppercase', letterSpacing:'0.08em' }}>iVect</span>
      </div>
      {[['(4,7)','(0,0,1,1,1,0)'],['(3,7)','(0,0,0,1,1,1)'],['(3,6)','(0,0,2,0,0,1)']].map(([a,b], i) => (
        <div key={i} style={{ display:'flex', alignItems:'center', gap:12, fontFamily:"'Raleway',sans-serif", fontWeight:700, fontSize:16, color:'#fff' }}>
          <span style={{ minWidth:60 }}>{a}</span>
          <svg width="32" height="12"><defs><marker id={`aw${i}`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="white"/></marker></defs><line x1="0" y1="6" x2="26" y2="6" stroke="white" strokeWidth="1.5" markerEnd={`url(#aw${i})`}/></svg>
          <span>{b}</span>
        </div>
      ))}
    </div>
  );
};

Object.assign(window, { SplitSlide, SplitDemoLeft, SplitDemoRight });
