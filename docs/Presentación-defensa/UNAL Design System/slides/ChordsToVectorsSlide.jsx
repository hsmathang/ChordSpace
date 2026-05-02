const MathText = ({ math, inline = true, style }) => {
  if (!window.katex) return <span>[KaTeX Loading...]</span>;
  const html = window.katex.renderToString(math, { throwOnError: false, displayMode: !inline });
  return <span style={style} dangerouslySetInnerHTML={{ __html: html }} />;
};

const ChordsToVectorsSlide = ({ pageNum = 16, department = "Facultad de Ciencias · Matemáticas Aplicadas" }) => {
  const step = window.useDeckStep(4, 'slide-chords-vectors');

  const rows = [
    { key: 'c', label: '\\text{C}=\\langle 0,4,7\\rangle', struct: '(4, 7)', y: 282 },
    { key: 'am', label: '\\text{Am}=\\langle 9,0,4\\rangle', struct: '(3, 7)', y: 432 },
    { key: 'bdim', label: '\\text{Bdim}=\\langle 11,2,5\\rangle', struct: '(3, 6)', y: 582 },
  ];

  return (
    <div style={{
      position: 'absolute',
      inset: 0,
      backgroundColor: 'white',
      overflow: 'hidden',
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      fontFamily: "'Raleway',sans-serif"
    }}>
      <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: '42%', backgroundColor: 'white' }} />
      <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: '58%', backgroundColor: '#E8610A' }} />

      <div style={{ position: 'relative', width: 1200, height: 800, zIndex: 10 }}>
        <h2 style={{ position: 'absolute', left: 48, top: 58, fontSize: 42, color: '#1A1A1A', margin: 0, fontWeight: 800 }}>
          Acordes como tuplas
        </h2>
        <div style={{ position: 'absolute', left: 52, top: 120, fontSize: 20, color: '#666', fontWeight: 600 }}>
          orden conservado
        </div>

        <h1 style={{
          position: 'absolute',
          left: 850,
          top: 56,
          transform: 'translateX(-50%)',
          color: 'white',
          fontSize: 50,
          fontWeight: 800,
          margin: 0,
          whiteSpace: 'nowrap'
        }}>
          PC-Set Theory
        </h1>

        <div style={{ position: 'absolute', left: 550, top: 168, color: 'white', fontSize: 34, fontWeight: 800 }}>
          iStruct
        </div>
        <div style={{ position: 'absolute', left: 795, top: 168, color: 'white', fontSize: 34, fontWeight: 800 }}>
          iVect
        </div>

        {rows.map((row) => (
          <div key={row.key} style={{ position: 'absolute', left: 50, top: row.y, fontSize: 32, color: '#1A1A1A' }}>
            <MathText math={row.label} />
          </div>
        ))}

        <div style={{ opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.6s' }}>
          {rows.map((row) => (
            <div key={`struct-${row.key}`} style={{
              position: 'absolute',
              left: 555,
              top: row.y,
              fontSize: 38,
              fontWeight: 800,
              color: 'white'
            }}>
              {row.struct}
            </div>
          ))}
        </div>

        <div style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.6s' }}>
          <div style={{
            position: 'absolute',
            left: 785,
            top: 350,
            fontSize: 34,
            fontWeight: 800,
            color: 'white',
            whiteSpace: 'nowrap'
          }}>
            (0, 0, 1, 1, 1, 0)
          </div>
          <div style={{
            position: 'absolute',
            left: 785,
            top: 580,
            fontSize: 34,
            fontWeight: 800,
            color: 'white',
            whiteSpace: 'nowrap'
          }}>
            (0, 0, 2, 0, 0, 1)
          </div>
        </div>

        <div style={{
          position: 'absolute',
          left: 765,
          top: 300,
          width: 390,
          height: 145,
          border: '3px solid rgba(255,255,255,0.9)',
          opacity: step >= 3 ? 1 : 0,
          transition: 'opacity 0.6s',
          pointerEvents: 'none'
        }} />
        <div style={{
          position: 'absolute',
          left: 790,
          top: 462,
          color: 'white',
          fontSize: 24,
          fontWeight: 700,
          opacity: step >= 3 ? 1 : 0,
          transition: 'opacity 0.6s'
        }}>
          misma huella clasica
        </div>

        <div style={{
          position: 'absolute',
          left: 38,
          bottom: 112,
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          opacity: step >= 4 ? 1 : 0,
          transform: step >= 4 ? 'translateY(0)' : 'translateY(28px)',
          transition: 'all 0.6s cubic-bezier(0.2, 0.8, 0.2, 1)'
        }}>
          <div style={{ position: 'relative', width: 86, height: 80 }}>
            <svg width="86" height="80" viewBox="0 0 105 96">
              <polygon points="52,5 5,91 100,91" fill="#F1C40F" stroke="#1A1A1A" strokeWidth="7" strokeLinejoin="round" />
              <text x="52" y="76" textAnchor="middle" fontSize="58" fontWeight="bold" fill="#C0392B" fontFamily="Arial">!</text>
            </svg>
          </div>
          <div style={{
            border: '2px solid #1A1A1A',
            backgroundColor: 'white',
            padding: '10px 18px',
            color: '#1A1A1A',
            fontSize: 24,
            fontWeight: 800,
            lineHeight: 1.2
          }}>
            clasifica<br />pero comprime orden
          </div>
        </div>

        <svg width="1200" height="800" viewBox="0 0 1200 800" style={{ position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 20 }}>
          <defs>
            <marker id="arrowBlackChordMap" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A1A1A" />
            </marker>
            <marker id="arrowWhiteChordMap" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#FFFFFF" />
            </marker>
          </defs>

          <g style={{ opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.6s' }}>
            <line x1="360" y1="306" x2="528" y2="306" stroke="#1A1A1A" strokeWidth="3" markerEnd="url(#arrowBlackChordMap)" />
            <line x1="378" y1="456" x2="528" y2="456" stroke="#1A1A1A" strokeWidth="3" markerEnd="url(#arrowBlackChordMap)" />
            <line x1="426" y1="606" x2="528" y2="606" stroke="#1A1A1A" strokeWidth="3" markerEnd="url(#arrowBlackChordMap)" />
          </g>

          <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.6s' }}>
            <path d="M 660 306 C 710 306 725 350 765 370" fill="none" stroke="#FFFFFF" strokeWidth="3" markerEnd="url(#arrowWhiteChordMap)" />
            <path d="M 660 456 C 710 456 725 405 765 390" fill="none" stroke="#FFFFFF" strokeWidth="3" markerEnd="url(#arrowWhiteChordMap)" />
            <line x1="660" y1="606" x2="765" y2="606" stroke="#FFFFFF" strokeWidth="3" markerEnd="url(#arrowWhiteChordMap)" />
          </g>

        </svg>
      </div>

      <InstitutionalFooter pageNum={pageNum} department={department} />
    </div>
  );
};

Object.assign(window, { ChordsToVectorsSlide });
