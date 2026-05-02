// TitleSlide.jsx — White slide with title, subtitle, department

const InstitutionalFooter = ({ pageNum, department = "Facultad de Ciencias · Matemáticas Aplicadas" }) => (
  <div style={{
    position: 'absolute', bottom: 0, left: 0, right: 0, height: 96,
    background: '#F0EBE0',
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 31px 0 34px',
    zIndex: 80,
    boxShadow: '0 -1px 0 rgba(27, 122, 62, 0.16)',
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: 16, minWidth: 0 }}>
      <div style={{ display: 'grid', gap: 6, flex: '0 0 auto' }}>
        <div style={{ width: 74, height: 4, background: '#2DC56E' }} />
        <div style={{ width: 43, height: 4, background: '#2DC56E' }} />
      </div>
      <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 16, color: '#1B7A3E', fontStyle: 'italic', lineHeight: 1.2, fontWeight: 700 }}>
        {department.split(/[·•]/).map((d, i) => <div key={i}>{d.trim()}</div>)}
      </div>
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 18, flex: '0 0 auto' }}>
      {pageNum !== undefined && (
        <div style={{ fontFamily: "'Raleway',sans-serif", fontSize: 17, color: '#1A1A1A', fontWeight: 700 }}>
          {pageNum}
        </div>
      )}
      <img
        src="assets/escudo-un-2016-footer.png"
        alt="Escudo Universidad Nacional de Colombia"
        style={{ width: 101, height: 89, objectFit: 'contain', mixBlendMode: 'multiply', opacity: 0.96 }}
      />
    </div>
  </div>
);

const SlideChrome = ({ pageNum = 2, department = "Facultad de Ciencias · Matemáticas Aplicadas", children }) => (
  <div style={{ position: 'absolute', inset: 0, background: '#fff', overflow: 'hidden' }}>
    {/* Page number */}
    <div style={{ position: 'absolute', top: 14, left: 18, fontSize: 13, fontFamily: "'Raleway',sans-serif", color: '#777' }}>{pageNum}</div>
    {/* Gold accent line top-right */}
    <div style={{ position: 'absolute', top: 16, right: 0, width: '58%', height: 2, background: '#E8A020' }} />
    {/* Content */}
    <div style={{ position: 'absolute', inset: '0 0 96px 0', padding: '52px 72px 0' }}>{children}</div>
    <InstitutionalFooter pageNum={pageNum} department={department} />
  </div>
);

const TitleSlide = ({
  pageNum = 2,
  title = "Computational Exploration of a Musical Chord Space",
  subtitle = "Una representación perceptual para acordes musicales",
  department = "Facultad de Ciencias · Matemáticas Aplicadas",
}) => (
  <div style={{position:'absolute',inset:0}}>
    <SlideChrome pageNum={pageNum} department={department}>
      <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', height: '100%', gap: 24 }}>
        {/* Title box */}
        <div style={{ border: '1px solid #ccc', padding: '20px 28px', display: 'inline-block' }}>
          <div style={{
            fontFamily: "'Playfair Display','Georgia',serif",
            fontWeight: 800,
            fontSize: 42,
            color: '#1B7A3E',
            lineHeight: 1.1,
            letterSpacing: '-0.01em',
          }}>{title}</div>
        </div>
        {/* Subtitle */}
        <div style={{
          borderTop: '2px solid #1B7A3E',
          borderBottom: '2px solid #1B7A3E',
          padding: '10px 0',
          textAlign: 'center',
        }}>
          <div style={{
            fontFamily: "'Playfair Display','Georgia',serif",
            fontWeight: 700,
            fontStyle: 'italic',
            fontSize: 20,
            color: '#1B7A3E',
            textDecoration: 'underline',
          }}>{subtitle}</div>
        </div>
        {/* Department */}
        <div style={{
          fontFamily: "'Raleway',sans-serif",
          fontWeight: 400,
          fontSize: 19,
          color: '#444',
          textAlign: 'center',
        }}>{department.replace('·', '–')}</div>
      </div>
    </SlideChrome>
  </div>
);

Object.assign(window, { InstitutionalFooter, SlideChrome, TitleSlide });
