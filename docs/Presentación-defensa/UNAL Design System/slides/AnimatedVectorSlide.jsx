const ArrayBox = ({ val, index, redBorderRight, textColor = '#ccc', stepTarget, currentStep, highlightRed }) => {
  return (
    <div style={{ 
      width: 80, height: 80, border: '3px solid #888', 
      borderRight: redBorderRight ? '4px solid #C0392B' : '3px solid #888',
      display: 'flex', justifyContent: 'center', alignItems: 'center',
      flexDirection: 'column', position: 'relative',
      marginLeft: -3,
      backgroundColor: 'white'
    }}>
      <div style={{ 
        fontSize: 50, fontWeight: 700, fontFamily: "'Raleway',sans-serif", 
        color: highlightRed ? '#C0392B' : textColor,
        transition: 'color 0.5s'
      }}>
        {val}
      </div>
      <div style={{ position: 'absolute', bottom: -35, fontSize: 22, fontWeight: 600, color: '#1A1A1A' }}>
        {index}
      </div>
    </div>
  );
};

const AnimatedVectorSlide = ({
  pageNum = 14,
  department = "Facultad de Ciencias · Matemáticas Aplicadas"
}) => {
  const step = window.useDeckStep(7, 'slide-diagram');

  // Data for arrays
  const arr11 = [
    { val: '0', color: '#ccc' }, // 1
    { val: '0', color: '#ccc' }, // 2
    { val: '1', color: '#27AE60', onStep: 4 }, // 3
    { val: '1', color: '#888', onStep: 4 }, // 4
    { val: '0', color: '#ccc' }, // 5
    { val: '0', color: '#ccc' }, // 6
    { val: '1', color: '#888', onStep: 4 }, // 7
    { val: '0', color: '#ccc' }, // 8
    { val: '0', color: '#ccc' }, // 9
    { val: '0', color: '#ccc' }, // 10
    { val: '0', color: '#ccc' }, // 11
  ];

  const arr6 = [
    { val: '0', color: '#ccc' }, // 1
    { val: '0', color: '#ccc' }, // 2
    { val: '1', color: '#27AE60' }, // 3
    { val: '1', color: '#888' }, // 4
    { val: '1', color: '#C0392B', onStep: 6 }, // 5 (Red)
    { val: '0', color: '#ccc' }, // 6
  ];

  return (
    <div style={{position:'absolute',inset:0}}>
      <SlideChrome pageNum={pageNum} department={department}>
        
        {/* We use a fixed logical coordinate system so absolute positioning works perfectly */}
        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
          
          {/* BIG "C" */}
          <div style={{ 
            position: 'absolute', top: 60, left: 200, 
            fontSize: 260, fontFamily: "'Raleway',sans-serif", fontWeight: 400, color: '#000',
            opacity: step >= 0 ? 1 : 0, transition: 'opacity 0.5s'
          }}>
            C
          </div>

          {/* Do Mi Sol Area */}
          <div style={{ position: 'absolute', top: 120, left: 650, opacity: step >= 0 ? 1 : 0, transition: 'opacity 0.5s' }}>
            <div style={{ display: 'flex', gap: 150, fontSize: 80, fontFamily: "'Raleway',sans-serif", fontWeight: 400, color: '#000' }}>
              <span>Do</span>
              <span>Mi</span>
              <span>Sol</span>
            </div>
            {/* Arcs for Do Mi Sol */}
            <svg style={{ position: 'absolute', top: -140, left: -60, width: 800, height: 400, pointerEvents: 'none' }}>
              <defs>
                <marker id="arrowGrey" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#666" />
                </marker>
              </defs>
              <g style={{ opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
                {/* Do to Mi (below) */}
                <path d="M 120,240 C 200,380 320,380 370,240" fill="none" stroke="#666" strokeWidth="3" markerEnd="url(#arrowGrey)" />
                <text x="235" y="380" fill="#888" fontSize="40" fontWeight="bold" fontFamily="Raleway">4</text>
                
                {/* Mi to Sol (below) */}
                <path d="M 400,240 C 480,380 570,380 620,240" fill="none" stroke="#666" strokeWidth="3" markerEnd="url(#arrowGrey)" />
                <text x="500" y="380" fill="#27AE60" fontSize="40" fontWeight="bold" fontFamily="Raleway">3</text>

                {/* Do to Sol (above) */}
                <path d="M 120,120 C 250,-20 480,-20 620,120" fill="none" stroke="#666" strokeWidth="3" markerEnd="url(#arrowGrey)" />
                <text x="350" y="20" fill="#888" fontSize="40" fontWeight="bold" fontFamily="Raleway">7</text>
              </g>
            </svg>
          </div>

          {/* Vector (4, 7, 3) */}
          <div style={{ 
            position: 'absolute', top: 320, left: 320, 
            fontSize: 70, fontFamily: "'Raleway',sans-serif", fontWeight: 700, color: '#888',
            opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s',
            zIndex: 20
          }}>
            (4, <span style={{ color: '#888' }}>7</span>, <span style={{ color: '#27AE60' }}>3</span>)
          </div>

          {/* Array 11 */}
          <div style={{ 
            display: 'flex', position: 'absolute', top: 550, left: 250, 
            opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s' 
          }}>
            {arr11.map((box, i) => {
              const showVal = step >= (box.onStep || 0);
              const showRedBorder = step >= 5 && i === 5; // Box 6 (index 5)
              return <ArrayBox key={i} val={showVal ? box.val : '0'} textColor={showVal ? box.color : '#ccc'} index={i+1} redBorderRight={showRedBorder} />;
            })}
          </div>

          {/* Array 6 */}
          <div style={{ 
            display: 'flex', position: 'absolute', top: 820, left: 250, 
            opacity: step >= 6 ? 1 : 0, transition: 'opacity 0.5s' 
          }}>
            {arr6.map((box, i) => {
              const showVal = step >= (box.onStep || 0);
              return <ArrayBox key={i} val={showVal ? box.val : '0'} textColor={showVal ? box.color : '#ccc'} index={i+1} highlightRed={i===4 && step>=6} />;
            })}
          </div>

          {/* Global SVG for mapping arrows */}
          <svg style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none', zIndex: 10 }}>
            <defs>
              <marker id="arrowGreen" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#27AE60" />
              </marker>
              <marker id="arrowGreyGlobal" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#888" />
              </marker>
              <marker id="arrowRed" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#C0392B" />
              </marker>
            </defs>

            {/* Step 4: Arrows from vector to array 11 */}
            <g style={{ opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              {/* Arrow 3 -> Box 3 (Green) */}
              {/* Box 3 center is at 250 + 80*2 + 40 = 450. Top of array is 550. */}
              {/* "3" in vector is around x=480, y=390. */}
              <line x1="490" y1="410" x2="450" y2="535" stroke="#27AE60" strokeWidth="3" markerEnd="url(#arrowGreen)" />

              {/* Arrow 7 -> Box 7 (Grey) */}
              {/* "7" is around x=430, y=340. Box 7 center is 250 + 80*6 + 40 = 770. */}
              <path d="M 430,340 C 430,150 770,150 770,535" fill="none" stroke="#888" strokeWidth="3" markerEnd="url(#arrowGreyGlobal)" />

              {/* Arrow 4 -> Box 4 (Grey) */}
              {/* "4" is around x=350, y=410. Box 4 center is 250 + 80*3 + 40 = 530. Bottom is 630. */}
              <path d="M 350,410 C 150,550 150,800 530,650" fill="none" stroke="#888" strokeWidth="3" markerEnd="url(#arrowGreyGlobal)" />
            </g>

            {/* Step 5 & 6: Red Operation Arrow */}
            <g style={{ opacity: step >= 5 ? 1 : 0, transition: 'opacity 0.8s ease' }}>
              {/* Starts at bottom of red line: x = 250 + 80*6 = 730, y = 630. */}
              {/* Ends at Box 5 of bottom array: x = 250 + 80*4 + 40 = 610, y = 820. */}
              <line x1="730" y1="650" x2="610" y2="805" stroke="#C0392B" strokeWidth="3" markerEnd="url(#arrowRed)" />
            </g>
          </svg>

          {/* Sombrero Bug (Step 7) */}
          <div style={{ 
            position: 'absolute', bottom: 80, right: 100, 
            opacity: step >= 7 ? 1 : 0, 
            transform: step >= 7 ? 'translateY(0)' : 'translateY(50px)',
            transition: 'all 0.8s cubic-bezier(0.2, 0.8, 0.2, 1)' 
          }}>
            <img src="assets/sombrero-bug.png" alt="Bug Mascot" style={{ width: 250 }} 
                 onError={(e) => { 
                   e.target.style.display = 'none'; 
                   e.target.nextSibling.style.display = 'block'; 
                 }} />
            {/* Fallback emoji if image is missing */}
            <div style={{ display: 'none', fontSize: 180 }}>🪲🤠</div>
          </div>

        </div>
      </SlideChrome>
    </div>
  );
};

Object.assign(window, { AnimatedVectorSlide });
