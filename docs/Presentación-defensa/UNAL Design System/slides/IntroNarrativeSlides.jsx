const PlayGlyph = ({ size = 24, color = '#1A1A1A' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
    <path d="M8 5 L19 12 L8 19 Z" fill={color} />
  </svg>
);

const playMelody = (midis, durations = null) => {
  if (!window.AudioEngine) return;
  let cursor = 0;
  midis.forEach((midi, i) => {
    const dur = durations?.[i] || 0.34;
    window.setTimeout(() => window.AudioEngine.playNote(midi, dur), cursor * 1000);
    cursor += dur * 0.88;
  });
};

const complementMelody = (midis) => {
  if (!midis.length) return [];
  const out = [midis[0]];
  for (let i = 1; i < midis.length; i += 1) {
    const d = midis[i] - midis[i - 1];
    if (d === 0) {
      out.push(out[out.length - 1]);
      continue;
    }
    const sign = d > 0 ? 1 : -1;
    const comp = sign * (12 - Math.abs(d % 12));
    out.push(out[out.length - 1] + comp);
  }
  return out;
};

const MelodyStaff = ({ notes, color = '#1B7A3E', muted = false }) => {
  const min = Math.min(...notes);
  const max = Math.max(...notes);
  const span = Math.max(max - min, 1);
  const points = notes.map((midi, i) => {
    const x = 48 + i * 88;
    const y = 150 - ((midi - min) / span) * 92;
    return { x, y, midi };
  });
  const d = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

  return (
    <svg width="720" height="190" viewBox="0 0 720 190" style={{ display: 'block' }}>
      {[0,1,2,3,4].map(i => (
        <line key={i} x1="24" x2="696" y1={42 + i * 24} y2={42 + i * 24} stroke="#D4D0C8" strokeWidth="1.4" />
      ))}
      <path d={d} fill="none" stroke={color} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" opacity={muted ? 0.42 : 1} />
      {points.map((p, i) => (
        <g key={`${p.midi}-${i}`}>
          <circle cx={p.x} cy={p.y} r="11" fill={color} opacity={muted ? 0.52 : 1} />
          <text x={p.x} y={p.y + 31} textAnchor="middle" fontSize="13" fontFamily="'JetBrains Mono',monospace" fill="#555">{p.midi}</text>
        </g>
      ))}
    </svg>
  );
};

const MiniVerticalChord = () => (
  <svg width="560" height="300" viewBox="0 0 560 300">
    <rect x="40" y="40" width="480" height="190" fill="#fff" stroke="#1A1A1A" strokeWidth="3" />
    {Array.from({ length: 12 }).map((_, i) => (
      <line key={i} x1={40 + i * 40} x2={40 + i * 40} y1="40" y2="230" stroke="#1A1A1A" strokeWidth={i % 7 === 0 ? 3 : 1.4} opacity={i % 7 === 0 ? 1 : 0.38} />
    ))}
    {[
      { x: 120, y: 176, label: 'Do', fill: '#C0392B' },
      { x: 280, y: 116, label: 'Mi', fill: '#2980B9' },
      { x: 400, y: 76, label: 'Sol', fill: '#27AE60' },
    ].map(n => (
      <g key={n.label}>
        <circle cx={n.x} cy={n.y} r="18" fill={n.fill} />
        <line x1={n.x} x2={n.x} y1={n.y} y2="250" stroke={n.fill} strokeWidth="2" strokeDasharray="5,6" />
        <text x={n.x} y="278" textAnchor="middle" fontSize="24" fontWeight="700" fontFamily="'Raleway',sans-serif" fill="#1A1A1A">{n.label}</text>
      </g>
    ))}
    <text x="280" y="28" textAnchor="middle" fontSize="20" fontWeight="700" fontFamily="'Raleway',sans-serif" fill="#1A1A1A">simultaneidad vertical</text>
  </svg>
);

const ChromaticBins = ({ mode = 'twelve' }) => {
  const labels = mode === 'six' ? ['1/11','2/10','3/9','4/8','5/7','6'] : ['1','2','3','4','5','6','7','8','9','10','11','12'];
  const n = labels.length;
  const radius = mode === 'six' ? 98 : 124;
  return (
    <svg width="360" height="360" viewBox="0 0 360 360">
      <circle cx="180" cy="180" r={radius} fill="#fff" stroke="#1A1A1A" strokeWidth="2" />
      {labels.map((label, i) => {
        const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
        const x = 180 + radius * Math.cos(angle);
        const y = 180 + radius * Math.sin(angle);
        const tx = 180 + (radius - 28) * Math.cos(angle);
        const ty = 180 + (radius - 28) * Math.sin(angle);
        const hot = label === '1' || label === '11' || label === '1/11';
        return (
          <g key={label}>
            <line x1="180" y1="180" x2={x} y2={y} stroke="#D4D0C8" strokeWidth="1.2" />
            <circle cx={tx} cy={ty} r={hot ? 19 : 15} fill={hot ? '#E8610A' : '#F0EBE0'} stroke="#1A1A1A" strokeWidth="1.4" />
            <text x={tx} y={ty + 6} textAnchor="middle" fontSize={hot ? 17 : 14} fontWeight="700" fontFamily="'Raleway',sans-serif" fill="#1A1A1A">{label}</text>
          </g>
        );
      })}
      <circle cx="180" cy="180" r="34" fill="#1A1A1A" />
    </svg>
  );
};

const ResearchQuestionSlide = ({ pageNum, department }) => (
  <div style={{ position: 'absolute', inset: 0 }}>
    <SlideChrome pageNum={pageNum} department={department}>
      <div style={{ height: '100%', display: 'grid', gridTemplateRows: '1fr auto', padding: '36px 74px 50px' }}>
        <div style={{ alignSelf: 'center', display: 'grid', gap: 34 }}>
          <div style={{
            fontFamily: "'Playfair Display','Georgia',serif",
            fontSize: 76,
            lineHeight: 1.04,
            fontWeight: 800,
            color: '#1A1A1A',
            maxWidth: 1040,
          }}>¿Cómo explorar acordes por su huella perceptual?</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 24, fontFamily: "'Raleway',sans-serif", fontSize: 23, fontWeight: 700, color: '#1B7A3E' }}>
            <span>acorde</span>
            <span style={{ width: 120, height: 3, background: '#E8A020' }} />
            <span>oído</span>
            <span style={{ width: 120, height: 3, background: '#E8A020' }} />
            <span>espacio navegable</span>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 7 }}>
          {Array.from({ length: 12 }).map((_, i) => (
            <div key={i} style={{ height: 42 + (i % 5) * 13, alignSelf: 'end', background: i % 3 === 0 ? '#1B7A3E' : i % 3 === 1 ? '#C0392B' : '#2980B9', opacity: 0.72, borderRadius: 2 }} />
          ))}
        </div>
      </div>
    </SlideChrome>
  </div>
);

const MusicalFocusSlide = ({ pageNum, department }) => (
  <div style={{ position: 'absolute', inset: 0 }}>
    <SlideChrome pageNum={pageNum} department={department}>
      <div style={{ height: '100%', padding: '44px 70px 40px', display: 'grid', gridTemplateColumns: '0.95fr 1.05fr', gap: 48, alignItems: 'center' }}>
        <div>
          <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 72, fontWeight: 800, lineHeight: 1.05, color: '#1A1A1A', marginBottom: 42 }}>
            La música tiene muchas capas
          </div>
          <div style={{ display: 'grid', gap: 18, fontFamily: "'Raleway',sans-serif" }}>
            {[
              ['Melodía', '#8A8A8A', 0.42],
              ['Ritmo', '#8A8A8A', 0.42],
              ['Armonía', '#1B7A3E', 1],
            ].map(([label, color, opacity]) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 18, opacity }}>
                <div style={{ width: 72, height: 5, background: color }} />
                <div style={{ fontSize: 42, fontWeight: 800, color }}>{label}</div>
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: 'grid', justifyItems: 'center', gap: 22 }}>
          <MiniVerticalChord />
          <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 24, lineHeight: 1.35, textAlign: 'center', color: '#444', maxWidth: 620 }}>
            En esta tesis, el acorde aislado es el objeto perceptual que queremos ubicar y comparar.
          </div>
        </div>
      </div>
    </SlideChrome>
  </div>
);

const ComplementIntervalsSlide = ({ pageNum, department }) => {
  const original = [
    67, 67, 67, 72, 76,
    67, 67, 67, 72, 76,
    72, 72, 71, 71, 69, 69, 67,
    65, 65, 65, 69, 72,
    67, 67, 67, 71, 74, 72,
  ];
  const durations = [
    0.22, 0.22, 0.22, 0.42, 0.62,
    0.22, 0.22, 0.22, 0.42, 0.62,
    0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.62,
    0.22, 0.22, 0.22, 0.42, 0.62,
    0.22, 0.22, 0.22, 0.42, 0.42, 0.72,
  ];
  const altered = complementMelody(original);
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <SlideChrome pageNum={pageNum} department={department}>
        <div style={{ height: '100%', padding: '28px 70px 36px', display: 'grid', gridTemplateRows: 'auto 1fr', gap: 12 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 60, fontWeight: 800, color: '#1A1A1A' }}>El oído no colapsa intervalos</div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button onClick={() => playMelody(original, durations)} style={{ width: 54, height: 54, borderRadius: '50%', border: '2px solid #1B7A3E', background: '#fff', display: 'grid', placeItems: 'center', cursor: 'pointer' }} title="Melodía original">
                <PlayGlyph color="#1B7A3E" />
              </button>
              <button onClick={() => playMelody(altered, durations)} style={{ width: 54, height: 54, borderRadius: '50%', border: '2px solid #C0392B', background: '#fff', display: 'grid', placeItems: 'center', cursor: 'pointer' }} title="Intervalos complementarios">
                <PlayGlyph color="#C0392B" />
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 42, alignItems: 'center' }}>
            <div style={{ display: 'grid', gap: 12 }}>
              <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 25, fontWeight: 800, color: '#1B7A3E' }}>melodía</div>
              <MelodyStaff notes={original} color="#1B7A3E" />
            </div>
            <div style={{ display: 'grid', gap: 12 }}>
              <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 25, fontWeight: 800, color: '#C0392B' }}>complementarios</div>
              <MelodyStaff notes={altered} color="#C0392B" />
            </div>
            <div style={{ gridColumn: '1 / span 2', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 18, marginTop: -8 }}>
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 26, color: '#1A1A1A' }}>+5</span>
              <span style={{ width: 180, height: 3, background: '#D4D0C8' }} />
              <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 26, color: '#C0392B' }}>+7</span>
              <span style={{ fontFamily: "'Raleway',sans-serif", fontSize: 22, color: '#555' }}>no cuentan la misma historia sonora</span>
            </div>
          </div>
        </div>
      </SlideChrome>
    </div>
  );
};

const TwelveBinsSlide = ({ pageNum, department }) => (
  <div style={{ position: 'absolute', inset: 0 }}>
    <SlideChrome pageNum={pageNum} department={department}>
      <div style={{ height: '100%', padding: '36px 70px 42px', display: 'grid', gridTemplateRows: 'auto 1fr', gap: 8 }}>
        <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 62, fontWeight: 800, color: '#1A1A1A' }}>Doce direcciones, no seis equivalencias</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 52, alignItems: 'center' }}>
          <div style={{ display: 'grid', justifyItems: 'center', gap: 16, opacity: 0.56 }}>
            <ChromaticBins mode="six" />
            <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 28, fontWeight: 800, color: '#555' }}>colapsar</div>
          </div>
          <div style={{ display: 'grid', justifyItems: 'center', gap: 16 }}>
            <ChromaticBins mode="twelve" />
            <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 28, fontWeight: 800, color: '#1B7A3E' }}>preservar</div>
          </div>
          <div style={{ gridColumn: '1 / span 2', justifySelf: 'center', display: 'flex', alignItems: 'center', gap: 20, fontFamily: "'Raleway',sans-serif", fontSize: 23, color: '#444' }}>
            <span style={{ color: '#E8610A', fontWeight: 800 }}>1</span>
            <span>y</span>
            <span style={{ color: '#E8610A', fontWeight: 800 }}>11</span>
            <span>pueden ser complementarios en papel, pero no equivalentes para la escucha.</span>
          </div>
        </div>
      </div>
    </SlideChrome>
  </div>
);

Object.assign(window, {
  ResearchQuestionSlide,
  MusicalFocusSlide,
  ComplementIntervalsSlide,
  TwelveBinsSlide,
});
