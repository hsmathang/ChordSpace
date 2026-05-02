const ArrayGrid = ({ length, startX, startY, values, colors, step, visibleStep }) => {
  const cellSize = 60;
  return (
    <g style={{ opacity: step >= visibleStep ? 1 : 0, transition: 'opacity 0.6s' }}>
      {Array.from({ length }).map((_, i) => {
        const idx = i + 1;
        const val = values[idx] || 0;
        const color = colors[idx] || (val === 0 ? '#CCCCCC' : '#777777');
        const x = startX + i * cellSize;
        const y = startY;
        
        return (
          <g key={`cell-${idx}`}>
            <rect 
              x={x} y={y} width={cellSize} height={cellSize} 
              fill="white" stroke="#1A1A1A" strokeWidth="1.5" 
            />
            <text 
              x={x + cellSize/2} y={y + 40} 
              fill={color} fontSize="32" fontWeight="bold" 
              fontFamily="'Raleway',sans-serif" textAnchor="middle"
            >
              {val}
            </text>
            <text 
              x={x + cellSize/2} y={y + cellSize + 20} 
              fill="#1A1A1A" fontSize="18" 
              fontFamily="'Raleway',sans-serif" textAnchor="middle"
            >
              {idx}
            </text>
          </g>
        );
      })}
    </g>
  );
};

const CMajorVectorSlide = ({ pageNum = 14 }) => {
  const step = window.useDeckStep(6, 'slide-c-major');

  return (
    <div style={{ position: 'absolute', inset: 0, backgroundColor: 'white', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      
      {/* Background SVG Canvas for precise absolute positioning */}
      <svg width="100%" height="100%" viewBox="0 0 1200 800" style={{ position: 'absolute', inset: 0 }}>
        
        <defs>
          <marker id="arrowGray" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#777777" />
          </marker>
          <marker id="arrowGreen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#27AE60" />
          </marker>
          <marker id="arrowRed" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#C0392B" />
          </marker>
          <marker id="arrowBlack" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A1A1A" />
          </marker>
        </defs>

        {/* Step 0: C Major Graph */}
        <text x="120" y="240" fill="#1A1A1A" fontSize="180" fontFamily="'Playfair Display',serif" fontWeight="800">C</text>
        
        <text x="500" y="180" fill="#1A1A1A" fontSize="56" fontFamily="'Raleway',sans-serif">Do</text>
        <text x="700" y="180" fill="#1A1A1A" fontSize="56" fontFamily="'Raleway',sans-serif">Mi</text>
        <text x="900" y="180" fill="#1A1A1A" fontSize="56" fontFamily="'Raleway',sans-serif">Sol</text>

        {/* Step 1: Interval distances in Graph */}
        <g style={{ opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.6s' }}>
          {/* Do -> Mi (4) */}
          <path d="M 540 200 Q 620 320 700 200" fill="none" stroke="#1A1A1A" strokeWidth="2.5" markerEnd="url(#arrowBlack)" />
          <text x="620" y="320" fill="#777777" fontSize="36" fontWeight="bold" fontFamily="'Raleway',sans-serif" textAnchor="middle">4</text>
          
          {/* Mi -> Sol (3) */}
          <path d="M 750 200 Q 840 320 920 200" fill="none" stroke="#1A1A1A" strokeWidth="2.5" markerEnd="url(#arrowBlack)" />
          <text x="835" y="320" fill="#27AE60" fontSize="36" fontWeight="bold" fontFamily="'Raleway',sans-serif" textAnchor="middle">3</text>
          
          {/* Do -> Sol (7) */}
          <path d="M 540 130 Q 750 30 920 130" fill="none" stroke="#1A1A1A" strokeWidth="2.5" markerEnd="url(#arrowBlack)" />
          <text x="740" y="55" fill="#777777" fontSize="36" fontWeight="bold" fontFamily="'Raleway',sans-serif" textAnchor="middle">7</text>
        </g>

        {/* Step 2: Tuple (4,7,3) */}
        <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.6s' }}>
          <text x="210" y="300" fill="#777777" fontSize="42" fontWeight="bold" fontFamily="'Raleway',sans-serif">(4, 7, </text>
          <text x="298" y="300" fill="#27AE60" fontSize="42" fontWeight="bold" fontFamily="'Raleway',sans-serif">3</text>
          <text x="323" y="300" fill="#777777" fontSize="42" fontWeight="bold" fontFamily="'Raleway',sans-serif">)</text>
        </g>

        {/* Step 3: 11-cell array and arrows for 3 and 4 */}
        <ArrayGrid 
          length={11} startX={150} startY={420} step={step} visibleStep={3}
          values={{ 3: 1, 4: 1, 7: step >= 4 ? 1 : 0 }} 
          colors={{ 3: '#27AE60', 4: '#777777', 7: '#777777' }} 
        />
        
        <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.6s' }}>
          {/* Arrow for 3 (Green) */}
          <line x1="305" y1="315" x2="305" y2="405" stroke="#27AE60" strokeWidth="2.5" markerEnd="url(#arrowGreen)" />
          
          {/* Arrow for 4 (Gray) */}
          <path d="M 235 320 Q 90 480 345 450" fill="none" stroke="#777777" strokeWidth="2.5" markerEnd="url(#arrowGray)" />
        </g>

        {/* Step 4: Arrow for 7 */}
        <g style={{ opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.6s' }}>
          <path d="M 275 260 Q 400 130 550 405" fill="none" stroke="#777777" strokeWidth="2.5" markerEnd="url(#arrowGray)" />
        </g>

        {/* Step 5: Red line and Mascot */}
        <g style={{ opacity: step >= 5 ? 1 : 0, transition: 'opacity 0.6s' }}>
          {/* Red line between 6 and 7 (X = 150 + 6*60 = 510) */}
          <line x1="510" y1="390" x2="510" y2="520" stroke="#C0392B" strokeWidth="3" />
        </g>

        {/* Step 6: Final 6-cell array and red arrow */}
        <ArrayGrid 
          length={6} startX={150} startY={620} step={step} visibleStep={6}
          values={{ 3: 1, 4: 1, 5: 1 }} 
          colors={{ 3: '#27AE60', 4: '#777777', 5: '#C0392B' }} 
        />
        
        <g style={{ opacity: step >= 6 ? 1 : 0, transition: 'opacity 0.6s' }}>
          {/* Arrow from 7 down to 5 */}
          {/* Cell 7 bottom is at 540, 480. Cell 5 top is at 420, 620 */}
          <path d="M 525 530 L 425 610" fill="none" stroke="#C0392B" strokeWidth="2.5" markerEnd="url(#arrowRed)" />
        </g>

      </svg>
      
      {/* HTML absolute overlays (Mascot Image) */}
      <img 
        src="assets/sombrero-bug.png" 
        alt="Sombrero Bug" 
        style={{
          position: 'absolute',
          right: 150,
          bottom: 120,
          width: 250,
          opacity: step >= 5 ? 1 : 0,
          transform: step >= 5 ? 'translateY(0) scale(1)' : 'translateY(40px) scale(0.9)',
          transition: 'all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)',
          pointerEvents: 'none'
        }}
      />

      {/* Footer Strip */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: 48, backgroundColor: '#F0EBE0', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 40px', zIndex: 10 }}>
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

Object.assign(window, { CMajorVectorSlide });
