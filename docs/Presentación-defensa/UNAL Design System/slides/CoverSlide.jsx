// CoverSlide.jsx — Universidad Nacional de Colombia
// Full green background with centered escudo + title

const CoverSlide = ({ title = "UNIVERSIDAD", subtitle = "NACIONAL DE COLOMBIA", department = "" }) => (
  <div style={{
    position: 'absolute', inset: 0,
    background: '#1B7A3E',
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 20,
    fontFamily: 'sans-serif',
  }}>
    {/* Top-right gold accent line */}
    <div style={{ position: 'absolute', top: 18, right: 0, width: '44%', height: 3, background: '#E8A020' }} />
    {/* Top-right green dot accent */}
    <div style={{ position: 'absolute', top: 18, right: '45%', width: '3%', height: 3, background: '#2DC56E' }} />

    <img
      src="assets/escudo-un-2016.jpg"
      alt="Escudo Universidad Nacional de Colombia"
      style={{
        width: 170,
        height: 170,
        objectFit: 'contain',
        filter: 'invert(1)',
        mixBlendMode: 'screen',
        opacity: 0.95
      }}
    />

    {/* Title */}
    <div style={{ textAlign: 'center' }}>
      <div style={{
        fontFamily: "'Playfair Display', 'Georgia', serif",
        fontWeight: 800,
        fontSize: 38,
        color: '#fff',
        letterSpacing: '0.06em',
        lineHeight: 1.1,
      }}>{title}</div>
      <div style={{
        fontFamily: "'Raleway', sans-serif",
        fontWeight: 700,
        fontSize: 17,
        color: 'rgba(255,255,255,0.9)',
        textTransform: 'uppercase',
        letterSpacing: '0.22em',
        marginTop: 8,
      }}>{subtitle}</div>
      {department && (
        <div style={{
          fontFamily: "'Raleway', sans-serif",
          fontWeight: 600,
          fontSize: 15,
          color: 'rgba(255,255,255,0.72)',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
          marginTop: 14,
        }}>{department}</div>
      )}
    </div>

    {/* Bottom-left green accent lines */}
    <div style={{ position: 'absolute', bottom: 36, left: 28, width: 52, height: 2, background: '#2DC56E' }} />
    <div style={{ position: 'absolute', bottom: 30, left: 28, width: 30, height: 2, background: '#2DC56E' }} />
  </div>
);

Object.assign(window, { CoverSlide });
