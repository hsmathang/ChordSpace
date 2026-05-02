// CoverSlide.jsx - sober thesis defense cover

const CoverSlide = ({
  title = "Modelo computacional para la exploracion de acordes en la composicion musical",
  author = "Hernan Santiago Angarita Garcia",
  program = "Maestria en Matematica Aplicada",
  director = "Director: Andres Torres",
  codirector = "Codirector: Francisco Gomez",
  university = "Universidad Nacional de Colombia",
  date = "2026",
}) => (
  <div style={{
    position: 'absolute', inset: 0,
    background: '#1B7A3E',
    display: 'grid',
    gridTemplateRows: '1fr auto',
    fontFamily: "'Raleway', sans-serif",
    color: '#fff',
    overflow: 'hidden',
  }}>
    <div style={{ position: 'absolute', top: 18, right: 0, width: '44%', height: 3, background: '#E8A020' }} />
    <div style={{ position: 'absolute', top: 18, right: '45%', width: '3%', height: 3, background: '#2DC56E' }} />

    <div style={{
      alignSelf: 'center',
      maxWidth: 980,
      marginLeft: 96,
      display: 'grid',
      gap: 30,
    }}>
      <div style={{
        fontFamily: "'Playfair Display', 'Georgia', serif",
        fontWeight: 800,
        fontSize: 68,
        lineHeight: 1.05,
        maxWidth: 940,
      }}>{title}</div>
      <div style={{
        fontWeight: 700,
        fontSize: 25,
        color: 'rgba(255,255,255,0.92)',
      }}>{author}</div>
    </div>

    <div style={{
      display: 'grid',
      gridTemplateColumns: '1.1fr 1fr',
      gap: 40,
      padding: '0 96px 72px',
      alignItems: 'end',
      fontSize: 18,
      lineHeight: 1.55,
      color: 'rgba(255,255,255,0.86)',
    }}>
      <div>
        <div style={{ fontWeight: 700 }}>{program}</div>
        <div>{university}</div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <div>{director}</div>
        <div>{codirector}</div>
        <div style={{ marginTop: 8, color: 'rgba(255,255,255,0.72)' }}>{date}</div>
      </div>
    </div>

    <div style={{ position: 'absolute', bottom: 36, left: 28, width: 52, height: 2, background: '#2DC56E' }} />
    <div style={{ position: 'absolute', bottom: 30, left: 28, width: 30, height: 2, background: '#2DC56E' }} />
  </div>
);

Object.assign(window, { CoverSlide });
