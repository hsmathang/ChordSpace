// ClosingSlide.jsx — Full green closing, italic centered text

const ClosingSlide = ({
  text = "Gracias",
  institution = "Universidad Nacional de Colombia",
}) => (
  <div style={{
    position: 'absolute', inset: 0,
    background: '#1B7A3E',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 32,
  }}>
    {/* Top accent */}
    <div style={{ position: 'absolute', top: 14, left: 0, right: 0, display: 'flex', justifyContent: 'space-between', padding: '0 0 0 16px' }}>
      <div style={{ width: 2, height: 2 }} />
      <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
        <div style={{ width: 40, height: 2, background: '#2DC56E' }} />
        <div style={{ width: 120, height: 2, background: '#E8A020' }} />
      </div>
    </div>

    {/* Main text */}
    <div style={{
      fontFamily: "'Playfair Display','Georgia',serif",
      fontWeight: 400,
      fontStyle: 'italic',
      fontSize: 72,
      color: '#fff',
      letterSpacing: '0.01em',
    }}>{text}</div>

    {/* Institution */}
    <div style={{
      fontFamily: "'Playfair Display','Georgia',serif",
      fontWeight: 400,
      fontStyle: 'italic',
      fontSize: 20,
      color: 'rgba(255,255,255,0.8)',
    }}>{institution}</div>

    {/* Bottom accent lines */}
    <div style={{ position: 'absolute', bottom: 30, left: 24, width: 52, height: 2, background: '#2DC56E' }} />
    <div style={{ position: 'absolute', bottom: 24, left: 24, width: 30, height: 2, background: '#2DC56E' }} />
  </div>
);

Object.assign(window, { ClosingSlide });
