const Reveal = ({ step, at = 1, children, x = 0, y = 18, scale = 1 }) => (
  <div style={{
    opacity: step >= at ? 1 : 0,
    transform: step >= at ? 'translate(0,0) scale(1)' : `translate(${x}px,${y}px) scale(${scale})`,
    transition: 'opacity 0.55s ease, transform 0.65s cubic-bezier(0.2,0.8,0.2,1)',
  }}>
    {children}
  </div>
);

const ArrowDefs = ({ id = 'narr' }) => (
  <defs>
    <marker id={`${id}-black`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1A1A1A" />
    </marker>
    <marker id={`${id}-green`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#1B7A3E" />
    </marker>
    <marker id={`${id}-orange`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#E8610A" />
    </marker>
  </defs>
);

const MiniPianoRoll = ({ step, compact = false }) => (
  <svg width={compact ? 260 : 420} height={compact ? 120 : 180} viewBox="0 0 420 180">
    <rect x="20" y="24" width="380" height="116" fill="#fff" stroke="#1A1A1A" strokeWidth="3" />
    {Array.from({ length: 12 }).map((_, i) => (
      <line key={i} x1={20 + i * 31.6} x2={20 + i * 31.6} y1="24" y2="140" stroke="#1A1A1A" strokeWidth={i % 7 === 0 ? 2.2 : 1} opacity={i % 7 === 0 ? 0.9 : 0.28} />
    ))}
    {[
      { x: 82, y: 110, c: '#C0392B', at: 1 },
      { x: 208, y: 72, c: '#2980B9', at: 2 },
      { x: 302, y: 46, c: '#27AE60', at: 3 },
    ].map((n, i) => (
      <g key={i} style={{ opacity: step >= n.at ? 1 : 0, transition: 'opacity 0.45s ease' }}>
        <circle cx={n.x} cy={n.y} r="15" fill={n.c} />
        <line x1={n.x} x2={n.x} y1={n.y} y2="154" stroke={n.c} strokeWidth="2" strokeDasharray="5,6" />
      </g>
    ))}
  </svg>
);

const MiniPitchWheel = ({ step, size = 250 }) => (
  <svg width={size} height={size} viewBox="0 0 130 130">
    <circle cx="65" cy="65" r="63" fill="#fff" stroke="#1A1A1A" strokeWidth="0.7"/>
    <circle cx="65" cy="65" r="40" fill="#fff" stroke="#1A1A1A" strokeWidth="0.7"/>
    <circle cx="65" cy="65" r="22" fill="#1A1A1A"/>
    {Array.from({ length: 12 }).map((_, i) => {
      const a = (i * 30 - 75) * Math.PI / 180;
      const t = (i * 30 - 90) * Math.PI / 180;
      return (
        <g key={i}>
          <path d={`M${65 + 40 * Math.cos(a)},${65 + 40 * Math.sin(a)} L${65 + 63 * Math.cos(a)},${65 + 63 * Math.sin(a)}`} stroke="#1A1A1A" strokeWidth="0.55"/>
          <text x={65 + 32 * Math.cos(t)} y={65 + 32 * Math.sin(t) + 3} textAnchor="middle" fontSize="10" fontFamily="'Raleway',sans-serif" fill="#1A1A1A">{i}</text>
        </g>
      );
    })}
    <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.45s ease' }}>
      <circle cx="65" cy="43" r="4" fill="#C0392B" />
      <circle cx="88" cy="65" r="4" fill="#2980B9" />
      <circle cx="48" cy="82" r="4" fill="#27AE60" />
    </g>
  </svg>
);

const ChordToken = ({ label, color = '#1A1A1A', small = false }) => (
  <div style={{
    border: '2px solid #1A1A1A',
    background: '#fff',
    padding: small ? '7px 10px' : '11px 14px',
    minWidth: small ? 72 : 96,
    textAlign: 'center',
    fontFamily: "'JetBrains Mono',monospace",
    fontSize: small ? 14 : 18,
    fontWeight: 800,
    color,
    boxShadow: small ? 'none' : '5px 5px 0 #E8E2D6',
  }}>{label}</div>
);

const MetricBox = ({ color, title, value, note }) => (
  <div style={{ border: '2px solid #1A1A1A', background: '#fff', padding: '10px 16px', minWidth: 160 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
      <span style={{ width: 26, height: 4, background: color, display: 'inline-block' }} />
      <span style={{ fontFamily: "'Raleway',sans-serif", fontSize: 18, fontWeight: 800, color: '#1A1A1A' }}>{title}</span>
    </div>
    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 26, fontWeight: 800, color }}>{value}</div>
    <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 13, color: '#555', marginTop: 4 }}>{note}</div>
  </div>
);

const Cloud = ({ step, count = 72 }) => (
  <svg width="520" height="390" viewBox="0 0 520 390">
    <rect x="18" y="18" width="484" height="344" fill="#F8F5EE" stroke="#1A1A1A" strokeWidth="2" />
    {Array.from({ length: count }).map((_, i) => {
      const x = 56 + ((i * 73) % 408);
      const y = 52 + ((i * 47) % 270);
      const r = 4 + (i % 5);
      const hot = i % 17 === 0;
      return <circle key={i} cx={x} cy={y} r={r} fill={hot ? '#E8610A' : (i % 3 === 0 ? '#1B7A3E' : '#2980B9')} opacity={step >= 2 ? (hot ? 0.95 : 0.42) : 0.08} />;
    })}
    <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s' }}>
      <circle cx="286" cy="190" r="72" fill="none" stroke="#C0392B" strokeWidth="4" strokeDasharray="10,8" />
      <text x="286" y="286" textAnchor="middle" fontSize="20" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#C0392B">zona cercana</text>
    </g>
  </svg>
);

const TwelveBars = ({ step, compact = false }) => {
  const vals = [28, 72, 44, 94, 36, 52, 81, 30, 58, 42, 75, 34];
  return (
    <div style={{ display: 'flex', alignItems: 'end', gap: compact ? 5 : 8, height: compact ? 95 : 138 }}>
      {vals.map((h, i) => (
        <div key={i} style={{
          width: compact ? 14 : 22,
          height: step >= 3 ? h : 8,
          background: i % 3 === 0 ? '#1B7A3E' : i % 3 === 1 ? '#C0392B' : '#2980B9',
          opacity: step >= 3 ? 0.78 : 0.22,
          transition: `height 0.55s ease ${i * 0.025}s, opacity 0.45s ease`,
        }} />
      ))}
    </div>
  );
};

const ExplorationProblemSlide = ({ pageNum, department }) => {
  const step = window.useDeckStep(5, 'slide-exploration-problem');
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ height: '100%', padding: '38px 70px 42px', display: 'grid', gridTemplateRows: 'auto 1fr', gap: 18 }}>
          <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 62, fontWeight: 800, color: '#1A1A1A' }}>El espacio es mucho más grande que el repertorio habitual</div>
          <div style={{ display: 'grid', gridTemplateColumns: '0.9fr 0.2fr 1.05fr', gap: 20, alignItems: 'center' }}>
            <Reveal step={step} at={1}>
              <div style={{ display: 'grid', gap: 18 }}>
                <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 28, fontWeight: 800, color: '#1B7A3E' }}>vocabulario conocido</div>
                <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', maxWidth: 430 }}>
                  {['C', 'Cm', 'G7', 'F', 'Am', 'Bdim', 'Csus4', 'Cmaj7'].map((c, i) => <ChordToken key={c} label={c} color={i % 2 ? '#1B7A3E' : '#1A1A1A'} />)}
                </div>
                <MiniPianoRoll step={step} compact />
              </div>
            </Reveal>
            <svg width="160" height="420" viewBox="0 0 160 420">
              <ArrowDefs id="explore" />
              <path d="M 26 205 C 78 118 78 302 132 205" fill="none" stroke="#1A1A1A" strokeWidth="4" markerEnd="url(#explore-black)" opacity={step >= 2 ? 1 : 0} style={{ transition: 'opacity 0.45s' }} />
              <text x="80" y="195" textAnchor="middle" fontSize="22" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#E8610A" opacity={step >= 2 ? 1 : 0}>generar</text>
            </svg>
            <Reveal step={step} at={2}>
              <div style={{ position: 'relative' }}>
                <Cloud step={step} />
                <Reveal step={step} at={4} y={24}>
                  <div style={{ position: 'absolute', right: 18, bottom: 8, display: 'flex', gap: 10 }}>
                    {['filtros', 'rango', 'cardinalidad', 'distancia'].map((t, i) => (
                      <div key={t} style={{ border: '2px solid #1A1A1A', background: i === 3 ? '#E8610A' : '#fff', color: i === 3 ? '#fff' : '#1A1A1A', padding: '8px 10px', fontSize: 15, fontWeight: 800, fontFamily: "'Raleway',sans-serif" }}>{t}</div>
                    ))}
                  </div>
                </Reveal>
              </div>
            </Reveal>
          </div>
        </div>
      </SlideChrome>
    </div>
  );
};

const ResearchQuestionAnimatedSlide = ({ pageNum, department }) => {
  const step = window.useDeckStep(6, 'slide-research-question-rich');
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ height: '100%', padding: '38px 72px 42px', position: 'relative' }}>
          <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 66, fontWeight: 800, lineHeight: 1.04, maxWidth: 1040, color: '#1A1A1A' }}>¿Cómo explorar acordes por su huella perceptual?</div>
          <svg width="1120" height="520" viewBox="0 0 1120 520" style={{ position: 'absolute', left: 72, top: 218 }}>
            <ArrowDefs id="rq" />
            <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
              <rect x="22" y="112" width="230" height="190" fill="#fff" stroke="#1A1A1A" strokeWidth="3" />
              <text x="137" y="92" textAnchor="middle" fontSize="24" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#1B7A3E">acorde</text>
            </g>
            <g transform="translate(15 120) scale(0.58)">
              <MiniPianoRoll step={step} />
            </g>
            <path d="M 266 206 C 330 120 390 120 452 206" fill="none" stroke="#1A1A1A" strokeWidth="4" markerEnd="url(#rq-black)" opacity={step >= 3 ? 1 : 0} style={{ transition: 'opacity 0.5s' }} />
            <g style={{ opacity: step >= 3 ? 1 : 0, transition: 'opacity 0.5s' }}>
              <rect x="470" y="120" width="210" height="176" fill="#F8F5EE" stroke="#1A1A1A" strokeWidth="3" />
              <text x="575" y="103" textAnchor="middle" fontSize="24" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#C0392B">oido</text>
              <path d="M 510 222 C 548 164 604 164 642 222" fill="none" stroke="#C0392B" strokeWidth="5" />
              <path d="M 518 242 C 552 204 600 204 634 242" fill="none" stroke="#2980B9" strokeWidth="4" />
              <path d="M 536 260 C 560 240 592 240 616 260" fill="none" stroke="#1B7A3E" strokeWidth="4" />
            </g>
            <path d="M 700 206 C 764 288 824 288 886 206" fill="none" stroke="#1B7A3E" strokeWidth="4" markerEnd="url(#rq-green)" opacity={step >= 4 ? 1 : 0} style={{ transition: 'opacity 0.5s' }} />
            <g style={{ opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.5s' }}>
              <rect x="910" y="88" width="184" height="246" fill="#fff" stroke="#1A1A1A" strokeWidth="3" />
              <text x="1002" y="67" textAnchor="middle" fontSize="24" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#1B7A3E">espacio navegable</text>
              {Array.from({ length: 32 }).map((_, i) => (
                <circle key={i} cx={940 + ((i * 37) % 124)} cy={120 + ((i * 53) % 178)} r={i % 7 === 0 ? 7 : 4} fill={i % 3 === 0 ? '#1B7A3E' : i % 3 === 1 ? '#C0392B' : '#2980B9'} opacity="0.62" />
              ))}
            </g>
          </svg>
          <Reveal step={step} at={5} y={20}>
            <div style={{ position: 'absolute', left: 170, bottom: 96, display: 'flex', alignItems: 'center', gap: 12, fontFamily: "'Raleway',sans-serif", fontSize: 20, fontWeight: 800 }}>
              {['estructura', 'rugosidad', 'distancia', 'sustitucion'].map((t, i) => (
                <React.Fragment key={t}>
                  <span style={{ border: '2px solid #1A1A1A', padding: '9px 13px', background: i === 1 ? '#E8610A' : '#fff', color: i === 1 ? '#fff' : '#1A1A1A' }}>{t}</span>
                  {i < 3 && <span style={{ width: 46, height: 3, background: '#1A1A1A' }} />}
                </React.Fragment>
              ))}
            </div>
          </Reveal>
        </div>
      </SlideChrome>
    </div>
  );
};

const HypothesisSubstitutionSlide = ({ pageNum, department }) => {
  const step = window.useDeckStep(6, 'slide-substitution-hypothesis');
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ height: '100%', padding: '36px 70px 42px', display: 'grid', gridTemplateRows: 'auto 1fr', gap: 12 }}>
          <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 62, lineHeight: 1.04, fontWeight: 800, color: '#1A1A1A' }}>La hipótesis: vecinos perceptuales</div>
          <div style={{ position: 'relative' }}>
            <svg width="100%" height="100%" viewBox="0 0 1160 660" style={{ position: 'absolute', inset: 0 }}>
              <ArrowDefs id="hyp" />
              <g style={{ opacity: step >= 1 ? 1 : 0, transition: 'opacity 0.5s' }}>
                <rect x="52" y="215" width="230" height="130" fill="#fff" stroke="#1A1A1A" strokeWidth="3" />
                <text x="167" y="196" textAnchor="middle" fontSize="24" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#1A1A1A">acorde conocido</text>
                <text x="167" y="293" textAnchor="middle" fontSize="52" fontWeight="800" fontFamily="'Playfair Display',serif" fill="#1A1A1A">C</text>
              </g>
              <path d="M 300 280 C 390 180 470 180 560 280" fill="none" stroke="#1A1A1A" strokeWidth="4" markerEnd="url(#hyp-black)" opacity={step >= 2 ? 1 : 0} style={{ transition: 'opacity 0.5s' }} />
              <g style={{ opacity: step >= 2 ? 1 : 0, transition: 'opacity 0.5s' }}>
                <rect x="592" y="78" width="500" height="440" fill="#F8F5EE" stroke="#1A1A1A" strokeWidth="3" />
                <circle cx="842" cy="300" r="130" fill="none" stroke="#E8610A" strokeWidth="4" strokeDasharray="12,8" />
                <text x="842" y="70" textAnchor="middle" fontSize="24" fontWeight="800" fontFamily="'Raleway',sans-serif" fill="#E8610A">region cercana</text>
                {Array.from({ length: 46 }).map((_, i) => (
                  <circle key={i} cx={632 + ((i * 57) % 420)} cy={120 + ((i * 71) % 348)} r={i % 11 === 0 ? 8 : 4} fill={i % 4 === 0 ? '#1B7A3E' : i % 4 === 1 ? '#C0392B' : '#2980B9'} opacity={step >= 3 ? 0.55 : 0.08} />
                ))}
              </g>
              <g style={{ opacity: step >= 4 ? 1 : 0, transition: 'opacity 0.5s' }}>
                <rect x="762" y="240" width="160" height="88" fill="#fff" stroke="#1A1A1A" strokeWidth="3" />
                <text x="842" y="292" textAnchor="middle" fontSize="32" fontWeight="800" fontFamily="'JetBrains Mono',monospace" fill="#1B7A3E">C+?</text>
                <path d="M 842 328 C 794 394 728 418 662 440" fill="none" stroke="#1B7A3E" strokeWidth="4" markerEnd="url(#hyp-green)" />
                <path d="M 842 328 C 906 396 970 404 1042 436" fill="none" stroke="#C0392B" strokeWidth="4" markerEnd="url(#hyp-orange)" />
              </g>
            </svg>
            <Reveal step={step} at={5} y={22}>
              <div style={{ position: 'absolute', left: 84, bottom: 78, display: 'flex', alignItems: 'center', gap: 18 }}>
                <MetricBox color="#1B7A3E" title="buscar" value="cercanos" note="sonoridad afin" />
                <MetricBox color="#C0392B" title="evitar" value="lejanos" note="contraste fuerte" />
                <MetricBox color="#E8610A" title="usar" value="distancia" note="perfil perceptual" />
              </div>
            </Reveal>
          </div>
        </div>
      </SlideChrome>
    </div>
  );
};

const RepresentationRequirementsSlide = ({ pageNum, department }) => {
  const step = window.useDeckStep(7, 'slide-representation-requirements');
  const items = [
    { t: 'preservar estructura', c: '#1A1A1A', icon: <MiniPitchWheel step={2} size={138} /> },
    { t: 'preservar percepcion', c: '#E8610A', icon: <TwelveBars step={3} compact /> },
    { t: 'permitir distancia', c: '#2980B9', icon: <svg width="150" height="100" viewBox="0 0 150 100"><line x1="25" y1="74" x2="124" y2="28" stroke="#2980B9" strokeWidth="5"/><circle cx="25" cy="74" r="11" fill="#1A1A1A"/><circle cx="124" cy="28" r="11" fill="#2980B9"/></svg> },
    { t: 'permitir exploracion', c: '#1B7A3E', icon: <svg width="150" height="100" viewBox="0 0 150 100">{Array.from({ length: 18 }).map((_, i) => <circle key={i} cx={20 + ((i*31)%110)} cy={18 + ((i*47)%68)} r={i%5===0?7:4} fill={i%2?'#1B7A3E':'#C0392B'} opacity="0.65" />)}</svg> },
  ];
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ height: '100%', padding: '34px 70px 42px', display: 'grid', gridTemplateRows: 'auto 1fr auto', gap: 18 }}>
          <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 60, lineHeight: 1.04, fontWeight: 800, color: '#1A1A1A' }}>Una buena representación debe hacer cuatro cosas</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 18, alignItems: 'stretch' }}>
            {items.map((item, i) => (
              <Reveal key={item.t} step={step} at={i + 1} y={32} scale={0.98}>
                <div style={{ height: 430, border: '3px solid #1A1A1A', background: i === 1 ? '#F8F5EE' : '#fff', padding: '18px 16px', display: 'grid', gridTemplateRows: '116px auto 1fr', gap: 14, boxShadow: '7px 7px 0 #E8E2D6' }}>
                  <div style={{ display: 'grid', placeItems: 'center' }}>{item.icon}</div>
                  <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 26, lineHeight: 1.08, fontWeight: 800, color: item.c }}>{item.t}</div>
                  <div style={{ display: 'grid', alignContent: 'end', gap: 8 }}>
                    {[0,1,2].map((_, j) => <div key={j} style={{ height: 10, width: `${68 + j * 24}%`, background: item.c, opacity: 0.22 + j * 0.18 }} />)}
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
          <Reveal step={step} at={5} y={18}>
            <div style={{ justifySelf: 'center', display: 'flex', alignItems: 'center', gap: 12, fontFamily: "'Raleway',sans-serif", fontSize: 20, fontWeight: 800 }}>
              <span style={{ border: '2px solid #1A1A1A', padding: '8px 12px', background: '#fff' }}>nombrar acordes</span>
              <span style={{ width: 50, height: 3, background: '#1A1A1A' }} />
              <span style={{ border: '2px solid #1A1A1A', padding: '8px 12px', background: '#E8610A', color: '#fff' }}>comparar acordes</span>
              <span style={{ width: 50, height: 3, background: '#1A1A1A' }} />
              <span style={{ border: '2px solid #1A1A1A', padding: '8px 12px', background: '#fff' }}>navegar posibilidades</span>
            </div>
          </Reveal>
        </div>
      </SlideChrome>
    </div>
  );
};

Object.assign(window, {
  ExplorationProblemSlide,
  ResearchQuestionAnimatedSlide,
  HypothesisSubstitutionSlide,
  RepresentationRequirementsSlide,
});
