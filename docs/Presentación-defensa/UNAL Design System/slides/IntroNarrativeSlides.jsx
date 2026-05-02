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

const MelodyStaff = ({ notes, durations = [], color = '#1B7A3E', muted = false }) => {
  const min = Math.min(...notes);
  const max = Math.max(...notes);
  const span = Math.max(max - min, 1);
  const points = notes.map((midi, i) => {
    const x = 40 + i * (640 / Math.max(notes.length - 1, 1));
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
          {i % 3 === 0 && <text x={p.x} y={p.y + 31} textAnchor="middle" fontSize="12" fontFamily="'JetBrains Mono',monospace" fill="#555">{p.midi}</text>}
          <rect x={p.x - 9} y="166" width="18" height={Math.max(5, (durations[i] || 0.22) * 22)} fill={color} opacity={muted ? 0.18 : 0.28} />
        </g>
      ))}
    </svg>
  );
};

const IntervalRibbon = ({ notes, color = '#1B7A3E' }) => {
  const intervals = notes.slice(1).map((n, i) => n - notes[i]);
  const sample = intervals.slice(0, 15);
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap', minHeight: 34 }}>
      {sample.map((d, i) => (
        <div key={`${d}-${i}`} style={{
          minWidth: 34,
          height: 28,
          display: 'grid',
          placeItems: 'center',
          border: '1.5px solid #1A1A1A',
          background: d === 0 ? '#F0EBE0' : '#fff',
          color: d === 0 ? '#777' : color,
          fontFamily: "'JetBrains Mono',monospace",
          fontSize: 13,
          fontWeight: 800,
        }}>
          {d > 0 ? `+${d}` : d}
        </div>
      ))}
    </div>
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

const CanonicalPitchClassWheel = ({ size = 500 }) => (
  <svg width={size} height={size} viewBox="0 0 130 130">
    <defs>
      <marker id="pilotArrowRedCirc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#C0392B" />
      </marker>
      <marker id="pilotArrowGreenCirc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#27AE60" />
      </marker>
      <marker id="pilotArrowBlueCirc" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#2980B9" />
      </marker>
    </defs>
    <circle cx="65" cy="65" r="64" fill="white" stroke="#1A1A1A" strokeWidth="0.5"/>
    <circle cx="65" cy="65" r="42" fill="white" stroke="#1A1A1A" strokeWidth="0.5"/>
    <circle cx="65" cy="65" r="24" fill="#1A1A1A"/>
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const a1 = (i * 30 - 75) * Math.PI / 180;
      return (
        <g key={i}>
          <path d={`M${65 + 42 * Math.cos(a1)},${65 + 42 * Math.sin(a1)} L${65 + 64 * Math.cos(a1)},${65 + 64 * Math.sin(a1)}`} stroke="#1A1A1A" strokeWidth="0.5"/>
          <path d={`M${65 + 24 * Math.cos(a1)},${65 + 24 * Math.sin(a1)} L${65 + 42 * Math.cos(a1)},${65 + 42 * Math.sin(a1)}`} stroke="#1A1A1A" strokeWidth="0.5"/>
        </g>
      );
    })}
    {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'].map((note, i) => {
      const angle = (i * 30 - 90) * Math.PI / 180;
      const x = 65 + 53 * Math.cos(angle);
      const y = 65 + 53 * Math.sin(angle);
      return <text key={note} x={x} y={y + 4} textAnchor="middle" fontSize="11" fontFamily="'Raleway',sans-serif" fill="#1A1A1A">{note}</text>;
    })}
    {[0,1,2,3,4,5,6,7,8,9,10,11].map(i => {
      const angle = (i * 30 - 90) * Math.PI / 180;
      return <text key={i} x={65 + 33 * Math.cos(angle)} y={65 + 33 * Math.sin(angle) + 3} textAnchor="middle" fontSize="11" fill="#1A1A1A" fontFamily="'Raleway',sans-serif">{i}</text>;
    })}
    <path d="M 65,37 A 28 28 0 0 1 76.83,39.62" fill="none" stroke="#C0392B" strokeWidth="1.8" markerEnd="url(#pilotArrowRedCirc)" />
    <path d="M 65,37 A 28 28 0 1 0 81.06,42.06" fill="none" stroke="#27AE60" strokeWidth="1.8" markerEnd="url(#pilotArrowGreenCirc)" />
    <path d="M 37,65 A 28 28 0 0 1 93,65" fill="none" stroke="#2980B9" strokeWidth="1.4" markerEnd="url(#pilotArrowBlueCirc)" opacity="0.72" />
    <circle cx="65" cy="37" r="2.8" fill="#C0392B" />
    <circle cx="79" cy="40.75" r="2.8" fill="#27AE60" />
  </svg>
);

const DirectionBox = ({ color, title, value, note }) => (
  <div style={{ border: '2px solid #1A1A1A', background: '#fff', padding: '10px 16px', minWidth: 160 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
      <span style={{ width: 26, height: 4, background: color, display: 'inline-block' }} />
      <span style={{ fontFamily: "'Raleway',sans-serif", fontSize: 18, fontWeight: 800, color: '#1A1A1A' }}>{title}</span>
    </div>
    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 26, fontWeight: 800, color }}>{value}</div>
    <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 13, color: '#555', marginTop: 4 }}>{note}</div>
  </div>
);

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
        <div style={{ height: '100%', padding: '26px 64px 36px', display: 'grid', gridTemplateRows: 'auto 1fr', gap: 10 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 58, fontWeight: 800, color: '#1A1A1A', lineHeight: 1.02 }}>El oído no colapsa intervalos</div>
              <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 10, fontFamily: "'Raleway',sans-serif", fontSize: 18, color: '#555', fontWeight: 700 }}>
                <span style={{ border: '2px solid #1B7A3E', padding: '4px 10px' }}>La Cucaracha</span>
                <span style={{ width: 42, height: 3, background: '#1A1A1A' }} />
                <span style={{ border: '2px solid #C0392B', padding: '4px 10px' }}>intervalos complementarios</span>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button onClick={() => playMelody(original, durations)} style={{ width: 54, height: 54, borderRadius: '50%', border: '2px solid #1B7A3E', background: '#fff', display: 'grid', placeItems: 'center', cursor: 'pointer' }} title="Melodía original">
                <PlayGlyph color="#1B7A3E" />
              </button>
              <button onClick={() => playMelody(altered, durations)} style={{ width: 54, height: 54, borderRadius: '50%', border: '2px solid #C0392B', background: '#fff', display: 'grid', placeItems: 'center', cursor: 'pointer' }} title="Intervalos complementarios">
                <PlayGlyph color="#C0392B" />
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateRows: '1fr 1fr auto', gap: 8, alignItems: 'center' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr 420px', gap: 18, alignItems: 'center' }}>
              <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 24, fontWeight: 800, color: '#1B7A3E' }}>melodía<br/><span style={{ fontSize: 14, color: '#555' }}>contorno reconocible</span></div>
              <MelodyStaff notes={original} durations={durations} color="#1B7A3E" />
              <IntervalRibbon notes={original} color="#1B7A3E" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '180px 1fr 420px', gap: 18, alignItems: 'center' }}>
              <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 24, fontWeight: 800, color: '#C0392B' }}>complementarios<br/><span style={{ fontSize: 14, color: '#555' }}>misma regla, otra escucha</span></div>
              <MelodyStaff notes={altered} durations={durations} color="#C0392B" />
              <IntervalRibbon notes={altered} color="#C0392B" />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: 18 }}>
              <div style={{ height: 2, background: '#D4D0C8' }} />
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 18, border: '2px solid #1A1A1A', background: '#fff', padding: '10px 18px' }}>
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 26, color: '#1A1A1A' }}>+5</span>
                <span style={{ width: 72, height: 3, background: '#D4D0C8' }} />
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 26, color: '#C0392B' }}>+7</span>
                <span style={{ fontFamily: "'Raleway',sans-serif", fontSize: 20, color: '#555' }}>no cuentan la misma historia sonora</span>
              </div>
              <div style={{ height: 2, background: '#D4D0C8' }} />
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
      <div style={{ height: '100%', padding: '34px 68px 42px', display: 'grid', gridTemplateRows: 'auto 1fr', gap: 8 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div style={{ fontFamily: "'Playfair Display','Georgia',serif", fontSize: 58, fontWeight: 800, color: '#1A1A1A', lineHeight: 1.04 }}>
            La rueda ya contiene<br />las dos direcciones
          </div>
          <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 20, color: '#666', maxWidth: 360, lineHeight: 1.35, paddingTop: 18 }}>
            No cambiamos de simbolo: enriquecemos la rueda PC-set que ya entiende el deck.
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '0.95fr 1.05fr', gap: 42, alignItems: 'center' }}>
          <div style={{ justifySelf: 'center', position: 'relative' }}>
            <CanonicalPitchClassWheel size={520} />
            <div style={{ position: 'absolute', top: 22, left: -22, border: '2px solid #C0392B', background: '#fff', padding: '6px 12px', fontFamily: "'JetBrains Mono',monospace", fontSize: 18, fontWeight: 800, color: '#C0392B' }}>+1</div>
            <div style={{ position: 'absolute', right: -24, bottom: 54, border: '2px solid #27AE60', background: '#fff', padding: '6px 12px', fontFamily: "'JetBrains Mono',monospace", fontSize: 18, fontWeight: 800, color: '#27AE60' }}>+11</div>
          </div>

          <div style={{ display: 'grid', gap: 18 }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
              <DirectionBox color="#C0392B" title="trayecto corto" value="C -> C#" note="semitono ascendente" />
              <DirectionBox color="#27AE60" title="trayecto largo" value="C -> B -> ... -> C#" note="complementario" />
            </div>
            <div style={{ border: '2px solid #1A1A1A', background: '#F8F5EE', padding: '18px 22px', display: 'grid', gap: 12 }}>
              <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 22, fontWeight: 800, color: '#1A1A1A' }}>La equivalencia circular sirve para nombrar clases de altura.</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 30, fontWeight: 800, color: '#C0392B' }}>1</div>
                <div style={{ height: 2, flex: 1, background: '#D4D0C8' }} />
                <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 30, fontWeight: 800, color: '#27AE60' }}>11</div>
              </div>
              <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 20, color: '#444', lineHeight: 1.32 }}>
                Pero para escuchar rugosidad, direccion y registro no son una decoracion: son informacion del objeto perceptual.
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontFamily: "'Raleway',sans-serif", fontSize: 18, fontWeight: 800 }}>
              {['clase', 'direccion', 'perfil 12D', 'rugosidad'].map((label, i) => (
                <React.Fragment key={label}>
                  <span style={{ border: '2px solid #1A1A1A', padding: '7px 11px', background: i === 2 ? '#E8610A' : '#fff', color: i === 2 ? '#fff' : '#1A1A1A' }}>{label}</span>
                  {i < 3 && <span style={{ width: 34, height: 3, background: '#1A1A1A' }} />}
                </React.Fragment>
              ))}
            </div>
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
