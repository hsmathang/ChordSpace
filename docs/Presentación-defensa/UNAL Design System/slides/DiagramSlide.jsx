// DiagramSlide.jsx — Schematic layout with zone columns and colored arrows

const DiagramSlide = ({
  pageNum = 6,
  title = "Esquemático",
  department = "Facultad de Ciencias · Matemáticas Aplicadas",
  children,
}) => (
  <div style={{position:'absolute',inset:0}}>
    <SlideChrome pageNum={pageNum} department={department}>
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div style={{
          fontFamily: "'Playfair Display','Georgia',serif",
          fontWeight: 800,
          fontSize: 30,
          color: '#1A1A1A',
          lineHeight: 1.1,
          textAlign: 'center',
        }}>{title}</div>
        <div style={{ flex: 1, position: 'relative' }}>{children}</div>
      </div>
    </SlideChrome>
  </div>
);

// Demo: Schematic flow with 4 zones
const SchematicDemo = () => {
  const zones = ['Codificación', 'Diversidad', 'Percepción', 'Visualización'];
  const rows = [
    { label: 'Vector Clásico', color: '#777', vals: ['(3,4,5)', 'Rugosidad', 'C'] },
    { label: 'Vector Propuesto 1', color: '#C0392B', vals: ['(4,7,3)', '→', 'C'] },
    { label: 'Vector Propuesto 2', color: '#1E6BB8', vals: ['(3,8,5)', '→', 'C/E'] },
    { label: 'Vector Propuesto 3', color: '#27AE60', vals: ['(5,9,4)', '→', 'G/S'] },
  ];

  return (
    <div style={{ position: 'relative', padding: '8px 0' }}>
      {/* Zone headers */}
      <div style={{ display: 'flex', borderBottom: '1px dashed #ccc', paddingBottom: 6, marginBottom: 12 }}>
        {zones.map(z => (
          <div key={z} style={{ flex: 1, borderRight: '1px dashed #ccc', paddingLeft: 8, lastChild: { borderRight: 'none' } }}>
            <span style={{ fontFamily: "'Raleway',sans-serif", fontWeight: 600, fontSize: 11, color: '#1E6BB8', textTransform: 'capitalize' }}>{z}</span>
          </div>
        ))}
      </div>

      {/* Flows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {rows.map((row, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
            <div style={{ flex: 1, paddingLeft: 8 }}>
              <div style={{ fontSize: 9, color: '#aaa', fontFamily: "'Raleway',sans-serif", marginBottom: 2 }}>{row.label}</div>
              <svg width="60" height="16">
                <defs><marker id={`fa${i}`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill={row.color}/></marker></defs>
                <line x1="0" y1="8" x2="54" y2="8" stroke={row.color} strokeWidth={i===0?1:2} strokeDasharray={i===0?"4,3":""} markerEnd={`url(#fa${i})`}/>
              </svg>
            </div>
            <div style={{ flex: 1, paddingLeft: 8 }}>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 15, fontWeight: 700, color: row.color }}>{row.vals[0]}</span>
            </div>
            <div style={{ flex: 1, paddingLeft: 8 }}>
              {i > 0 && <div style={{ width: 80, height: 48, background: '#e8e8e8', border: '1px solid #ccc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: 9, color: '#888', fontFamily: "'Raleway',sans-serif" }}>Rugosidad</span>
              </div>}
              {i === 0 && <span style={{ fontSize: 12, color: '#888', fontFamily: "'Raleway',sans-serif", fontStyle: 'italic' }}>Rugosidad</span>}
            </div>
            <div style={{ flex: 1, paddingLeft: 8 }}>
              <div style={{ width: 40, height: 40, border: '1px solid #ccc', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ fontSize: 10, fontFamily: "'Raleway',sans-serif", color: i===0?'#888':row.color, fontWeight: i>0?700:400 }}>{row.vals[2]}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

Object.assign(window, { DiagramSlide, SchematicDemo });
