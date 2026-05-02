const MathText = ({ math, inline = true, style }) => {
  if (!window.katex) return <span>[KaTeX Loading...]</span>;
  const html = window.katex.renderToString(math, { throwOnError: false, displayMode: !inline });
  return <span style={style} dangerouslySetInnerHTML={{ __html: html }} />;
};

const ChromaticCircleSVG = ({ step, size = 420 }) => (
  <svg width={size} height={size} viewBox="0 0 130 130">
    <circle cx="65" cy="65" r="64" fill="white" stroke="#1A1A1A" strokeWidth="0.5"/>
    <circle cx="65" cy="65" r="42" fill="white" stroke="#1A1A1A" strokeWidth="0.5"/>
    <circle cx="65" cy="65" r="24" fill="#1A1A1A"/>
    
    {/* Outer division lines */}
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const a1 = (i*30-75)*Math.PI/180; 
      const r1=42, r2=64, cx=65, cy=65;
      return <path key={`out-${i}`} d={`M${cx+r1*Math.cos(a1)},${cy+r1*Math.sin(a1)} L${cx+r2*Math.cos(a1)},${cy+r2*Math.sin(a1)}`}
        stroke="#1A1A1A" strokeWidth="0.5"/>;
    })}

    {/* Inner division lines */}
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const a1 = (i*30-75)*Math.PI/180; 
      const r1=24, r2=42, cx=65, cy=65;
      return <path key={`in-${i}`} d={`M${cx+r1*Math.cos(a1)},${cy+r1*Math.sin(a1)} L${cx+r2*Math.cos(a1)},${cy+r2*Math.sin(a1)}`}
        stroke="#1A1A1A" strokeWidth="0.5"/>;
    })}

    {/* Note texts (Chromatic, C=0 is top) */}
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
        textAnchor="middle" fontSize="11" fill="#1A1A1A" fontFamily="'Raleway',sans-serif">{i}</text>;
    })}

    {/* Inner arrows (step 5) */}
    <g style={{ opacity: step >= 5 ? 1 : 0, transition: 'opacity 0.5s' }}>
      <defs>
        <marker id="arrowRedCirc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#C0392B" />
        </marker>
        <marker id="arrowGreenCirc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="#27AE60" />
        </marker>
      </defs>

      {/* Red arrow from 0 to 1 (Clockwise, short route = 1) */}
      <path d="M 65,37 A 28 28 0 0 1 76.83,39.62" fill="none" stroke="#C0392B" strokeWidth="1.5" markerEnd="url(#arrowRedCirc)" />
      
      {/* Green arrow from 0 to 1 (Counter-clockwise, long route = 11) */}
      <path d="M 65,37 A 28 28 0 1 0 81.06,42.06" fill="none" stroke="#27AE60" strokeWidth="1.5" markerEnd="url(#arrowGreenCirc)" />
      
      {/* Dots at 0 and 1 */}
      <circle cx="65" cy="37" r="2.5" fill="#C0392B" />
      <circle cx="79" cy="40.75" r="2.5" fill="#27AE60" />
    </g>
  </svg>
);

const WarningTriangle = ({ step }) => (
  <div style={{
    position: 'absolute', bottom: 40, left: 240,
    opacity: step >= 7 ? 1 : 0,
    transform: step >= 7 ? 'scale(1) rotate(0deg)' : 'scale(0) rotate(-45deg)',
    transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)'
  }}>
    <svg width="400" height="340" viewBox="0 0 300 260">
      <polygon points="150,10 10,250 290,250" fill="#F1C40F" stroke="#1A1A1A" strokeWidth="20" strokeLinejoin="round" />
      <text x="150" y="200" textAnchor="middle" fontSize="160" fontWeight="bold" fill="#C0392B" fontFamily="Arial">!</text>
    </svg>
    <img src="assets/sombrero-bug.png" alt="Bug" style={{ 
      position: 'absolute', right: 40, bottom: 40, width: 140,
    }} onError={(e) => { e.target.style.display='none'; e.target.nextSibling.style.display='block'; }} />
    <div style={{ display: 'none', position: 'absolute', right: 40, bottom: 30, fontSize: 90 }}>🪲</div>
  </div>
);

const AnimatedPCSetSlide = ({ pageNum = 15 }) => {
  const step = window.useDeckStep(7, 'slide-split');

  return (
    <div style={{ position: 'absolute', inset: 0, backgroundColor: 'white', display: 'flex', overflow: 'hidden' }}>
      
      {/* Left White Panel */}
      <div style={{ flex: '6.5', position: 'relative', padding: '100px 80px', fontSize: 36, fontFamily: "'Raleway',sans-serif", color: '#1A1A1A', lineHeight: 1.4 }}>
         
         {/* Intro Text */}
         <div>
           Sea <MathText math="\mathcal{P}(\mathbb{Z}_{12})" /> el conjunto potencia de <MathText math="\mathbb{Z}_{12}" />, donde cada elemento <MathText math="x, y \in \mathcal{P}(\mathbb{Z}_{12})" /> representa un conjunto de clases de altura, es decir, un acorde.
         </div>

         {/* Bullets */}
         <div style={{ display: 'flex', flexDirection: 'column', gap: 30, marginTop: 50 }}>
           
           <div style={{ display: 'flex', gap: 20, opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.5s' }}>
             <div style={{ color: '#1E6BB8', fontWeight: 'bold' }}>❶</div>
             <div>
               <b>Relación de Octava:</b> Dos notas <MathText math="x_i" /> y <MathText math="x_j" /> están en relación de octava si <MathText math="x_i \sim_{oct} x_j \Leftrightarrow f(x_i) = 2^k f(x_j)" /> para algún <MathText math="k \in \mathbb{Z}" />.
             </div>
           </div>

           <div style={{ display: 'flex', gap: 20, opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
             <div style={{ color: '#1E6BB8', fontWeight: 'bold' }}>❷</div>
             <div>
               <b>Transposición <MathText math="T_n" />:</b> <MathText math="x \sim_T y \Leftrightarrow \exists n" /> tal que <MathText math="T_n(x) = y" />.
             </div>
           </div>

           <div style={{ 
             display: 'flex', gap: 20, padding: 10, marginLeft: -14,
             border: step >= 3 ? '4px solid #C0392B' : '4px solid transparent',
             opacity: step >= 3 ? 1 : 0, transition: 'all 0.5s ease',
             transform: step >= 3 ? 'scale(1)' : 'scale(0.98)'
           }}>
             <div style={{ color: '#1E6BB8', fontWeight: 'bold' }}>❸</div>
             <div>
               <b>Permutaciones de Tonos:</b> <MathText math="x \sim_\sigma y \Leftrightarrow \exists \sigma \in S_n" /> tal que <MathText math="\sigma(x) = y" />.
             </div>
           </div>
         </div>

         <WarningTriangle step={step} />

         {/* Formula Box */}
         <div style={{
           position: 'absolute', bottom: 120, right: -180, zIndex: 30,
           backgroundColor: 'white', padding: '15px 40px',
           border: '2px solid #1A1A1A',
           fontSize: 50, fontFamily: "'Raleway',sans-serif",
           opacity: step >= 6 ? 1 : 0,
           transform: step >= 6 ? 'translateY(0)' : 'translateY(50px)',
           transition: 'all 0.6s cubic-bezier(0.2, 0.8, 0.2, 1)'
         }}>
           <MathText math="d(x_i, y_j) = \min(|x_i - y_j|, 12 - |x_i - y_j|)" />
         </div>

      </div>

      {/* Right Orange Panel */}
      <div style={{ flex: '3.5', backgroundColor: '#E8610A', display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: 80, borderLeft: '4px solid #1A1A1A' }}>
         <h1 style={{ color: 'white', fontFamily: "'Raleway',sans-serif", fontSize: 65, fontWeight: 700, margin: 0 }}>
           PC-Set Theory
         </h1>

         <div style={{ 
           marginTop: 120, 
           opacity: step >= 4 ? 1 : 0, 
           transform: step >= 4 ? 'scale(1)' : 'scale(0.8)',
           transition: 'all 0.8s cubic-bezier(0.2, 0.8, 0.2, 1)'
         }}>
           <ChromaticCircleSVG step={step} size={500} />
         </div>
      </div>

    </div>
  );
};

Object.assign(window, { AnimatedPCSetSlide });
