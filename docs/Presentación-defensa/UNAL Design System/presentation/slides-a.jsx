// slides-a.jsx — Diapositivas 1–22
// Apertura + Marco teórico

const { useState, useEffect, useRef } = React;

// ─── Shared title/label helpers ────────────────────────────────
const GT = ({children,size=52,color=C.green,italic=false,style={}}) => (
  <div style={{fontFamily:"'Playfair Display',serif",fontWeight:800,fontSize:size,
    color,lineHeight:1.1,fontStyle:italic?'italic':'normal',...style}}>{children}</div>
);
const Sub = ({children,size=15,color=C.green,style={}}) => (
  <div style={{fontFamily:"'Raleway',sans-serif",fontWeight:700,fontSize:size,
    textTransform:'uppercase',letterSpacing:'0.15em',color,...style}}>{children}</div>
);
const Label = ({children,color=C.black,size=18,style={}}) => (
  <div style={{fontFamily:"'Raleway',sans-serif",fontSize:size,color,...style}}>{children}</div>
);
const Eq = ({children,size=18,style={}}) => (
  <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:size,
    border:'1px dashed #555',padding:'8px 18px',display:'inline-block',...style}}>{children}</div>
);
const Box = ({children,color=C.green,style={}}) => (
  <div style={{border:`1.5px solid ${color}`,padding:'14px 20px',...style}}>{children}</div>
);

// ─── Slide 1: Portada ───────────────────────────────────────────
const S1 = ({step}) => (
  <div style={{width:1280,height:720,position:'relative',background:C.green,overflow:'hidden',
    display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:20}}>
    <div style={{position:'absolute',top:16,right:0,width:'42%',height:3,background:C.gold}}/>
    <div style={{position:'absolute',top:16,right:'43%',width:'4%',height:3,background:C.greenL}}/>
    {/* Escudo placeholder */}
    <svg width="70" height="70" viewBox="0 0 52 52" fill="none">
      <circle cx="26" cy="26" r="22" stroke="white" strokeWidth="1.5" opacity="0.7"/>
      <rect x="18" y="13" width="16" height="18" stroke="white" strokeWidth="1.2" opacity="0.7"/>
      <path d="M18 25 Q26 31 34 25" stroke="white" strokeWidth="1.1" opacity="0.6" fill="none"/>
      <text x="26" y="23" textAnchor="middle" fontSize="8" fill="white" opacity="0.8" fontFamily="serif">π</text>
    </svg>
    <div style={{textAlign:'center',maxWidth:860}}>
      <GT size={38} color="#fff" style={{marginBottom:8}}>
        Modelo computacional para la exploración perceptual<br/>de acordes en la composición musical
      </GT>
      <div style={{width:200,height:2,background:C.gold,margin:'18px auto'}}/>
      <Label color="rgba(255,255,255,0.85)" size={16} style={{marginBottom:6}}>
        Hernán Santiago Angarita García
      </Label>
      <Label color="rgba(255,255,255,0.7)" size={13}>
        Maestría en Matemática Aplicada — Universidad Nacional de Colombia
      </Label>
      <Label color="rgba(255,255,255,0.65)" size={12} style={{marginTop:6}}>
        Director: Andrés Torres &nbsp;·&nbsp; Codirector: Francisco Gómez
      </Label>
    </div>
    <div style={{position:'absolute',bottom:30,left:24,width:56,height:2,background:C.greenL}}/>
    <div style={{position:'absolute',bottom:24,left:24,width:32,height:2,background:C.greenL}}/>
  </div>
);

// ─── Slide 2: Una pregunta ──────────────────────────────────────
const S2 = ({step}) => (
  <SlideChrome num={2}>
    <div style={{height:'100%',display:'flex',alignItems:'center',justifyContent:'center'}}>
      <Appear at={1} step={step}>
        <GT size={58} italic style={{textAlign:'center',maxWidth:880,color:C.green}}>
          ¿Cómo explorar acordes por su huella perceptual?
        </GT>
      </Appear>
    </div>
  </SlideChrome>
);

// ─── Slide 3: Por qué importa hoy ──────────────────────────────
const S3 = ({step}) => (
  <SlideChrome num={3}>
    <GT size={36}>Por qué importa hoy</GT>
    <Row gap={60} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      {[
        {at:1, label:'IA Musical', sub:'generación, recomendación'},
        {at:2, label:'Percepción', sub:'psicoacústica computacional'},
        {at:3, label:'Exploración', sub:'nuevas sonoridades posibles'},
      ].map(({at,label,sub})=>(
        <Appear key={at} at={at} step={step}>
          <Box color={C.green} style={{width:220,textAlign:'center',padding:'24px 20px'}}>
            <GT size={22} color={C.green}>{label}</GT>
            <Sub size={11} style={{marginTop:8,letterSpacing:'0.1em'}}>{sub}</Sub>
          </Box>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 4: Música y foco ─────────────────────────────────────
const S4 = ({step}) => (
  <SlideChrome num={4}>
    <GT size={36}>Foco del trabajo</GT>
    <Row gap={0} align="stretch" justify="center" style={{height:'calc(100% - 90px)',gap:0}}>
      {[
        {at:1,label:'Melodía',active:false},
        {at:2,label:'Ritmo',active:false},
        {at:3,label:'Armonía',active:true,sub:'el acorde aislado'},
      ].map(({at,label,active,sub})=>(
        <Appear key={at} at={at} step={step} style={{flex:1}}>
          <div style={{
            flex:1,height:'100%',display:'flex',flexDirection:'column',
            alignItems:'center',justifyContent:'center',gap:12,
            background:active?C.green:'transparent',
            border:`2px solid ${active?C.green:'#ddd'}`,
            margin:'0 8px',padding:'30px 20px',
          }}>
            <GT size={32} color={active?'#fff':C.gray}>{label}</GT>
            {sub&&<Sub size={13} color={active?C.gold:'#aaa'} style={{letterSpacing:'0.12em'}}>{sub}</Sub>}
          </div>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 5: Qué es un acorde ─────────────────────────────────
const S5 = ({step}) => {
  const C3=48,E3=52,G3=55;
  const dotsMap = step>=2 ? {[C3]:{color:C.red},[E3]:{color:C.blue},[G3]:{color:C.dgreen}} :
                  step>=1 ? {[C3]:{color:C.red}} : {};
  const handlePlay = () => AudioEngine.playChord([C3,E3,G3]);
  return(
    <SlideChrome num={5}>
      <Row gap={20} align="center">
        <GT size={36}>El acorde</GT>
        {step>=1&&<SoundBtn onClick={()=>AudioEngine.playChord([C3,E3,G3])}/>}
      </Row>
      <div style={{height:'calc(100% - 90px)',display:'flex',flexDirection:'column',
        alignItems:'center',justifyContent:'center',gap:24}}>
        <Appear at={1} step={step}>
          <PianoKeyboard startNote={48} endNote={60} dots={dotsMap} W={42}
            noteLabels={step>=3?{
              [C3]:{text:'0',color:C.red},
              [E3]:{text:'4',color:C.blue},
              [G3]:{text:'7',color:C.dgreen},
            }:{}}
            onPlay={n=>AudioEngine.playNote(n)}
          />
        </Appear>
        {step>=2&&(
          <Appear at={2} step={step}>
            <Eq size={20}>C_M = {'{48, 52, 55}'}</Eq>
          </Appear>
        )}
      </div>
    </SlideChrome>
  );
};

// ─── Slide 6: El problema de exploración ───────────────────────
const S6 = ({step}) => (
  <SlideChrome num={6}>
    <GT size={36}>El espacio de acordes posibles</GT>
    <Row gap={60} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      <Appear at={1} step={step}>
        <Col gap={10} align="center">
          <Sub size={12} color={C.gray}>Vocabulario habitual</Sub>
          {['C_M','a_m','G7','F_M','d_m','B°'].map(ch=>(
            <Label key={ch} size={18} color={C.green} style={{fontFamily:"'JetBrains Mono',monospace"}}>{ch}</Label>
          ))}
        </Col>
      </Appear>
      <Appear at={1} step={step}>
        <ArrowH color={C.gold} length={80}/>
      </Appear>
      <Appear at={2} step={step}>
        <Col gap={8} align="center">
          <Sub size={12} color={C.gray}>Espacio combinatorio</Sub>
          <div style={{fontSize:72,color:'#ddd',lineHeight:1}}>{'{ ··· }'}</div>
          <Label size={14} color={C.gray} style={{textAlign:'center',maxWidth:200}}>
            miles de estructuras<br/>sin organización perceptual
          </Label>
        </Col>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 7: Pregunta de investigación ────────────────────────
const S7 = ({step}) => (
  <SlideChrome num={7}>
    <div style={{height:'100%',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:30}}>
      <Appear at={1} step={step} style={{textAlign:'center',maxWidth:900}}>
        <GT size={32} italic>
          ¿Es posible construir un espacio de representación para acordes aislados,
          basado en su huella psicoacústica de rugosidad,
          que permita ubicar y descubrir acordes con sonoridades afines?
        </GT>
      </Appear>
    </div>
  </SlideChrome>
);

// ─── Slide 8: Hipótesis ─────────────────────────────────────────
const S8 = ({step}) => (
  <SlideChrome num={8}>
    <GT size={36}>Hipótesis de trabajo</GT>
    <div style={{height:'calc(100% - 90px)',display:'flex',alignItems:'center',justifyContent:'center'}}>
      <svg width={860} height={300} style={{overflow:'visible'}}>
        {/* Acorde consulta */}
        <rect x={0} y={110} width={160} height={60} fill="none" stroke={C.green} strokeWidth={2}/>
        <text x={80} y={137} textAnchor="middle" fontSize="16" fontFamily="Raleway,sans-serif" fill={C.green} fontWeight="700">sus4</text>
        <text x={80} y={158} textAnchor="middle" fontSize="11" fontFamily="Raleway,sans-serif" fill={C.gray}>acorde consulta</text>
        {/* Arrow */}
        {step>=1&&<>
          <line x1={162} y1={140} x2={240} y2={140} stroke={C.gold} strokeWidth={2} markerEnd="url(#arH)"/>
          <defs><marker id="arH" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0 L0,7 L7,3.5 Z" fill={C.gold}/></marker></defs>
        </>}
        {/* Space region */}
        {step>=1&&<>
          <ellipse cx={450} cy={140} rx={160} ry={110} fill={C.green+'11'} stroke={C.green} strokeWidth={1.5} strokeDasharray="4,3"/>
          <text x={450} y={97} textAnchor="middle" fontSize="12" fontFamily="Raleway,sans-serif" fill={C.green}>espacio Φ</text>
        </>}
        {/* Neighbors */}
        {step>=2&&[
          {x:390,y:100,label:'sus2',c:C.blue},
          {x:510,y:100,label:'7sus4',c:C.blue},
          {x:360,y:170,label:'add9',c:C.dgreen},
          {x:540,y:170,label:'M9',c:C.dgreen},
        ].map(({x,y,label,c})=>(
          <g key={label}>
            <circle cx={x} cy={y} r={22} fill="none" stroke={c} strokeWidth={1.5}/>
            <text x={x} y={y+4} textAnchor="middle" fontSize="11" fontFamily="Raleway,sans-serif" fill={c} fontWeight="700">{label}</text>
          </g>
        ))}
      </svg>
    </div>
  </SlideChrome>
);

// ─── Slide 9: Requisitos ────────────────────────────────────────
const S9 = ({step}) => (
  <SlideChrome num={9}>
    <GT size={36}>Una buena representación debe…</GT>
    <Col gap={22} style={{marginTop:30}}>
      {[
        {at:1,icon:'⬡',text:'Preservar la estructura interna del acorde'},
        {at:2,icon:'◉',text:'Capturar su huella perceptual'},
        {at:3,icon:'↔',text:'Permitir medir distancias entre acordes'},
        {at:4,icon:'↗',text:'Permitir explorar regiones desconocidas'},
      ].map(({at,icon,text})=>(
        <Appear key={at} at={at} step={step}>
          <Row gap={20} align="center">
            <div style={{fontSize:28,width:36,textAlign:'center',color:C.gold}}>{icon}</div>
            <GT size={26} color={C.black}>{text}</GT>
          </Row>
        </Appear>
      ))}
    </Col>
  </SlideChrome>
);

// ─── Slide 10: Modelos existentes ──────────────────────────────
const S10 = ({step}) => (
  <SlideChrome num={10}>
    <GT size={36}>Modelos existentes</GT>
    <Row gap={0} align="stretch" justify="center" style={{height:'calc(100% - 90px)',gap:2}}>
      {[
        {at:1,title:'Algebraicos',sub:'Forte, PC-Set Theory',note:'clasifican bien,\ncomprimen mucho'},
        {at:2,title:'Geométricos',sub:'Tymoczko, orbifolds',note:'relacionan acordes,\ncriterio: voice-leading'},
        {at:3,title:'Datos / IA',sub:'embeddings, corpus',note:'aprenden estilo,\npoca interpretabilidad'},
      ].map(({at,title,sub,note})=>(
        <Appear key={at} at={at} step={step} style={{flex:1,padding:'0 4px'}}>
          <div style={{border:`1.5px solid #ddd`,height:'100%',padding:'32px 24px',
            display:'flex',flexDirection:'column',gap:12}}>
            <GT size={24} color={C.green}>{title}</GT>
            <Sub size={12} color={C.gray} style={{letterSpacing:'0.08em'}}>{sub}</Sub>
            <div style={{flex:1}}/>
            <Label size={15} color={C.gray} style={{borderTop:'1px solid #eee',paddingTop:12,
              whiteSpace:'pre-line',fontStyle:'italic'}}>{note}</Label>
          </div>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 11: Helmholtz ────────────────────────────────────────
const S11 = ({step}) => (
  <SlideChrome num={11}>
    <Row gap={60} align="center" style={{height:'100%'}}>
      <Col gap={16}>
        <GT size={54}>Helmholtz</GT>
        <Sub size={14} color={C.gray} style={{letterSpacing:'0.1em'}}>On the Sensations of Tone</Sub>
        <Appear at={1} step={step}>
          <GT size={22} color={C.black} style={{maxWidth:420,marginTop:16}}>
            La disonancia tiene base <span style={{color:C.green}}>física y fisiológica</span>,<br/>no solo cultural.
          </GT>
        </Appear>
      </Col>
      <Appear at={1} step={step}>
        <svg width={300} height={200} style={{overflow:'visible'}}>
          {[0,1].map(i=>{
            const freq=i===0?2:2.15; const amp=40;
            const pts=Array.from({length:100},(_,x)=>`${x*3},${100+amp*Math.sin((x/100)*freq*2*Math.PI)}`).join(' ');
            return<polyline key={i} points={pts} fill="none" stroke={i===0?C.blue:C.red} strokeWidth="2"/>;
          })}
          <text x="150" y="180" textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif">batimientos → rugosidad</text>
        </svg>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 12: Batimientos animados ────────────────────────────
const S12 = ({step}) => {
  const [t, setT] = useState(0);
  const rafRef = useRef();
  useEffect(()=>{
    const loop = () => { setT(p=>p+0.02); rafRef.current=requestAnimationFrame(loop); };
    rafRef.current = requestAnimationFrame(loop);
    return ()=>cancelAnimationFrame(rafRef.current);
  },[]);
  const W=640,H=80,pts1=[],pts2=[],ptsSum=[];
  for(let x=0;x<W;x++){
    const f1=Math.sin((x/W)*8*Math.PI+t);
    const f2=Math.sin((x/W)*8.6*Math.PI+t);
    pts1.push(`${x},${H/2-f1*28}`);
    pts2.push(`${x},${H/2-f2*28}`);
    ptsSum.push(`${x},${H/2-(f1+f2)*18}`);
  }
  return(
    <SlideChrome num={12}>
      <GT size={36}>Batimientos y rugosidad</GT>
      <Col gap={8} style={{marginTop:16}}>
        <Appear at={1} step={step}>
          <Sub size={11} color={C.blue} style={{letterSpacing:'0.1em'}}>Tono A — f₁</Sub>
          <svg width={W} height={H}><polyline points={pts1.join(' ')} fill="none" stroke={C.blue} strokeWidth="2"/></svg>
        </Appear>
        <Appear at={2} step={step}>
          <Sub size={11} color={C.red} style={{letterSpacing:'0.1em'}}>Tono B — f₂ ≈ f₁</Sub>
          <svg width={W} height={H}><polyline points={pts2.join(' ')} fill="none" stroke={C.red} strokeWidth="2"/></svg>
        </Appear>
        <Appear at={3} step={step}>
          <Sub size={11} color={C.orange} style={{letterSpacing:'0.1em'}}>Interferencia → rugosidad sensorial</Sub>
          <svg width={W} height={H}><polyline points={ptsSum.join(' ')} fill="none" stroke={C.orange} strokeWidth="2.5"/></svg>
        </Appear>
      </Col>
    </SlideChrome>
  );
};

// ─── Slide 13: Plomp–Levelt ─────────────────────────────────────
const S13 = ({step}) => {
  const W=560,H=220;
  const curve = Array.from({length:100},(_,i)=>{
    const x=i/99; const r=x*Math.exp(1-x/0.25)*0.25;
    return [i*(W/99), H-r*H*0.92];
  });
  return(
    <SlideChrome num={13}>
      <Row gap={60} align="center" style={{height:'100%'}}>
        <Col gap={12}>
          <GT size={42}>Plomp & Levelt</GT>
          <Sub size={13} color={C.gray} style={{letterSpacing:'0.08em'}}>1965</Sub>
          <Appear at={2} step={step}>
            <GT size={20} color={C.black} style={{maxWidth:320,marginTop:10}}>
              Máxima rugosidad:<br/><span style={{color:C.green}}>¼ de banda crítica</span>
            </GT>
          </Appear>
        </Col>
        <Appear at={1} step={step}>
          <svg width={W} height={H+40} style={{overflow:'visible'}}>
            {/* Axes */}
            <line x1="0" y1={H} x2={W} y2={H} stroke="#bbb" strokeWidth="1.5"/>
            <line x1="0" y1="0" x2="0" y2={H} stroke="#bbb" strokeWidth="1.5"/>
            {/* Curve */}
            {step>=2&&<polyline points={curve.map(([x,y])=>`${x},${y}`).join(' ')}
              fill="none" stroke={C.orange} strokeWidth="3"/>}
            {/* Labels */}
            <text x={W/2} y={H+28} textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif">Separación / Banda crítica</text>
            <text x={-8} y={H/2} textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif" transform={`rotate(-90,-8,${H/2})`}>Rugosidad</text>
            {/* Peak marker */}
            {step>=2&&<>
              <line x1={W*0.25} y1={0} x2={W*0.25} y2={H} stroke={C.green} strokeWidth="1.5" strokeDasharray="4,3"/>
              <text x={W*0.25} y={-6} textAnchor="middle" fontSize="11" fill={C.green} fontFamily="Raleway,sans-serif">¼ BC</text>
            </>}
          </svg>
        </Appear>
      </Row>
    </SlideChrome>
  );
};

// ─── Slide 14: Cóclea ───────────────────────────────────────────
const S14 = ({step}) => (
  <SlideChrome num={14}>
    <Row gap={60} align="center" style={{height:'100%'}}>
      <Col gap={12}>
        <GT size={42}>Cóclea</GT>
        <GT size={22} color={C.black} style={{maxWidth:320}}>
          La rugosidad tiene base <span style={{color:C.green}}>fisiológica</span>:<br/>no depende solo del aprendizaje cultural.
        </GT>
      </Col>
      <Appear at={1} step={step}>
        <svg width={380} height={250} style={{overflow:'visible'}}>
          {/* Simplified cochlea spiral */}
          {Array.from({length:120},(_,i)=>{
            const t=i/120*3.5*Math.PI; const r=80-i*0.55;
            const x=190+r*Math.cos(t), y=130+r*Math.sin(t)*0.5;
            return {x,y,i};
          }).map(({x,y,i},_,arr)=>{
            if(i===0)return null;
            const prev=arr[i-1];
            return<line key={i} x1={prev.x} y1={prev.y} x2={x} y2={y} stroke={C.blue} strokeWidth="4" strokeLinecap="round" opacity={0.5+i*0.004}/>;
          })}
          <text x={190} y={246} textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif">membrana basilar</text>
          {/* frequency gradient */}
          <text x={80} y={80} textAnchor="middle" fontSize="10" fill={C.red} fontFamily="Raleway,sans-serif">alta f</text>
          <text x={270} y={150} textAnchor="middle" fontSize="10" fill={C.dgreen} fontFamily="Raleway,sans-serif">baja f</text>
        </svg>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 15: Cultura vs fisiología ───────────────────────────
const S15 = ({step}) => (
  <SlideChrome num={15}>
    <GT size={36}>Cultura y fisiología</GT>
    <Row gap={40} align="stretch" justify="center" style={{height:'calc(100% - 90px)',gap:40}}>
      {[
        {at:1,title:'Rugosidad sensorial',color:C.green,text:'Fenómeno fisiológico.\nEstable entre culturas.'},
        {at:2,title:'Preferencia cultural',color:C.orange,text:'Varía entre grupos\ny contextos históricos.'},
      ].map(({at,title,color,text})=>(
        <Appear key={at} at={at} step={step} style={{flex:1}}>
          <div style={{border:`2px solid ${color}`,height:'100%',
            display:'flex',flexDirection:'column',alignItems:'center',
            justifyContent:'center',gap:16,padding:'30px 24px'}}>
            <GT size={24} color={color}>{title}</GT>
            <Label size={18} color={C.black} style={{textAlign:'center',whiteSpace:'pre-line'}}>{text}</Label>
          </div>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 16: McDermott y los Tsimané ─────────────────────────
const S16 = ({step}) => (
  <SlideChrome num={16}>
    <Row gap={60} align="center" style={{height:'100%'}}>
      <Col gap={16}>
        <GT size={40}>McDermott</GT>
        <Sub size={12} color={C.gray} style={{letterSpacing:'0.08em'}}>Tsimané, Amazonía — 2016</Sub>
        <Appear at={1} step={step}>
          <GT size={20} color={C.black} style={{maxWidth:380,marginTop:10}}>
            Preferencia por consonancia:<br/><span style={{color:C.orange}}>no universal</span><br/><br/>
            Rugosidad como fenómeno sensorial:<br/><span style={{color:C.green}}>sustrato fisiológico estable</span>
          </GT>
        </Appear>
      </Col>
      <Appear at={1} step={step}>
        <svg width={280} height={260}>
          {/* Simple world map silhouette → just a box placeholder */}
          <rect x={10} y={10} width={260} height={160} fill="none" stroke="#ddd" strokeWidth="1.5"/>
          <text x={140} y={100} textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif">Amazonia</text>
          <circle cx={90} cy={95} r={8} fill={C.orange}/>
          <text x={140} y={200} textAnchor="middle" fontSize="13" fill={C.green} fontFamily="Raleway,sans-serif" fontWeight="700">
            rugosidad ≠ preferencia estética
          </text>
        </svg>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 17: Sethares ─────────────────────────────────────────
const S17 = ({step}) => (
  <SlideChrome num={17}>
    <Row gap={60} align="center" style={{height:'100%'}}>
      <Col gap={12}>
        <GT size={50}>Sethares</GT>
        <Sub size={13} color={C.gray} style={{letterSpacing:'0.08em'}}>Tuning, Timbre, Spectrum, Scale — 1993/2005</Sub>
        <Appear at={1} step={step}>
          <GT size={22} color={C.black} style={{maxWidth:380,marginTop:16}}>
            Consonancia = función de<br/>
            <span style={{color:C.green}}>intervalo</span> <span style={{color:C.black}}>+</span> <span style={{color:C.orange}}>timbre</span>
          </GT>
        </Appear>
      </Col>
      <Appear at={1} step={step}>
        <svg width={300} height={220} style={{overflow:'visible'}}>
          <rect x={10} y={20} width={100} height={50} fill="none" stroke={C.blue} strokeWidth="1.5"/>
          <text x={60} y={50} textAnchor="middle" fontSize="13" fill={C.blue} fontFamily="Raleway,sans-serif" fontWeight="700">Nota A</text>
          <rect x={10} y={90} width={100} height={50} fill="none" stroke={C.red} strokeWidth="1.5"/>
          <text x={60} y={120} textAnchor="middle" fontSize="13" fill={C.red} fontFamily="Raleway,sans-serif" fontWeight="700">Nota B</text>
          <line x1={112} y1={45} x2={160} y2={130} stroke={C.gray} strokeWidth="1.5" strokeDasharray="3,2"/>
          <line x1={112} y1={115} x2={160} y2={130} stroke={C.gray} strokeWidth="1.5" strokeDasharray="3,2"/>
          <rect x={160} y={110} width={120} height={40} fill={C.orange+'22'} stroke={C.orange} strokeWidth="1.5"/>
          <text x={220} y={134} textAnchor="middle" fontSize="13" fill={C.orange} fontFamily="Raleway,sans-serif" fontWeight="700">Rugosidad</text>
          <text x={220} y={185} textAnchor="middle" fontSize="11" fill={C.gray} fontFamily="Raleway,sans-serif">parciales × parciales</text>
        </svg>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 18: Qué toma la tesis de Sethares ───────────────────
const S18 = ({step}) => (
  <SlideChrome num={18}>
    <GT size={36}>Modelo de rugosidad heredado</GT>
    <Col gap={24} style={{marginTop:20,alignItems:'center'}}>
      <Appear at={1} step={step}>
        <Row gap={12} align="center">
          {['f₁','f₂','f₃','f₄','f₅','f₆'].map((f,i)=>(
            <div key={i} style={{textAlign:'center'}}>
              <div style={{width:30,background:C.blue,
                height:Math.round(80*Math.pow(0.88,i)),
                display:'flex',alignItems:'flex-end',justifyContent:'center'}}>
              </div>
              <Label size={11} color={C.blue} style={{marginTop:4}}>{f}</Label>
            </div>
          ))}
          <Label size={18} color={C.gray} style={{marginLeft:12}}>× 2 notas = pares de parciales</Label>
        </Row>
      </Appear>
      <Appear at={2} step={step}>
        <ArrowH color={C.gold} length={60} label="suma"/>
      </Appear>
      <Appear at={2} step={step}>
        <Eq size={20}>R(n_i, n_j) = Σ d(f_a, f_b, A_a, A_b)</Eq>
      </Appear>
      <Appear at={3} step={step}>
        <Row gap={30} align="center">
          <Label size={16} color={C.gray}>H = 6 armónicos</Label>
          <Label size={16} color={C.gray}>δ = 0.88</Label>
          <Label size={16} color={C.gray}>b₁ = 3.5, b₂ = 5.75</Label>
        </Row>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 19: Allen Forte ─────────────────────────────────────
const S19 = ({step}) => (
  <SlideChrome num={19}>
    <Row gap={60} align="center" style={{height:'100%'}}>
      <Col gap={12}>
        <GT size={50}>Allen Forte</GT>
        <Sub size={13} color={C.gray} style={{letterSpacing:'0.08em'}}>The Structure of Atonal Music — 1973</Sub>
        <Appear at={2} step={step}>
          <GT size={20} color={C.green} style={{maxWidth:320,marginTop:12}}>
            Catálogo finito y navegable.<br/>Lenguaje taxonómico fuerte.
          </GT>
        </Appear>
      </Col>
      <Appear at={1} step={step}>
        <Col gap={16} align="center">
          <Eq size={16}>C_M → {'{0, 4, 7}'}</Eq>
          <ArrowH color={C.gray} length={50} label="mod 12"/>
          <Eq size={16}>pitch-class set: 3-11</Eq>
          <ArrowH color={C.gray} length={50} label="forma prima"/>
          <Eq size={16}>[0, 3, 7]</Eq>
        </Col>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 20: Lo que Forte sacrifica ─────────────────────────
const S20 = ({step}) => (
  <SlideChrome num={20}>
    <GT size={36}>Lo que se pierde con Forte</GT>
    <Row gap={40} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      {[
        {at:1,midi:[36,40,43],W:26,label:'C₂ M\n(grave)',text:'rugosidad alta',color:C.red},
        {at:1,midi:[48,52,55],W:26,label:'C₃ M\n(medio)',text:'rugosidad media',color:C.orange},
        {at:1,midi:[60,64,67],W:26,label:'C₄ M\n(agudo)',text:'rugosidad baja',color:C.dgreen},
      ].map(({at,midi,W:kW,label,text,color},i)=>(
        <Appear key={i} at={at} step={step}>
          <Col gap={10} align="center">
            <PianoKeyboard startNote={midi[0]-2} endNote={midi[midi.length-1]+2}
              dots={Object.fromEntries(midi.map(n=>[n,{color}]))}
              W={kW} showMidi={false} showSolfege={false}/>
            <Eq size={13} style={{padding:'4px 10px'}}>[0, 4, 7]</Eq>
            <Label size={13} color={color} style={{textAlign:'center',whiteSpace:'pre-line'}}>{text}</Label>
          </Col>
        </Appear>
      ))}
      <Appear at={2} step={step} style={{position:'absolute',bottom:70,left:0,right:0,textAlign:'center'}}>
        <GT size={20} italic color={C.orange}>mismo PC-set ≠ misma percepción</GT>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 21: Tymoczko ────────────────────────────────────────
const S21 = ({step}) => (
  <SlideChrome num={21}>
    <Row gap={60} align="center" style={{height:'100%'}}>
      <Col gap={12}>
        <GT size={50}>Tymoczko</GT>
        <Sub size={13} color={C.gray} style={{letterSpacing:'0.08em'}}>A Geometry of Music — 2011</Sub>
        <Appear at={2} step={step}>
          <GT size={20} color={C.black} style={{maxWidth:380,marginTop:12}}>
            Organización geométrica: voice-leading,<br/>orbifolds, conducción de voces.
          </GT>
        </Appear>
      </Col>
      <Appear at={1} step={step}>
        <svg width={280} height={240}>
          {/* Simplified orbifold sketch */}
          {[
            {x:80,y:60,label:'C_M',c:C.green},{x:200,y:60,label:'a_m',c:C.blue},
            {x:140,y:150,label:'e_m',c:C.dgreen},{x:60,y:190,label:'F_M',c:C.green},
            {x:220,y:190,label:'G_M',c:C.green},
          ].map(({x,y,label,c})=>(
            <g key={label}>
              <circle cx={x} cy={y} r={22} fill="none" stroke={c} strokeWidth="1.5"/>
              <text x={x} y={y+5} textAnchor="middle" fontSize="12" fill={c} fontFamily="Raleway,sans-serif" fontWeight="700">{label}</text>
            </g>
          ))}
          {[[80,60,200,60],[200,60,140,150],[80,60,140,150],[140,150,60,190],[140,150,220,190]].map(([x1,y1,x2,y2],i)=>(
            <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#ccc" strokeWidth="1"/>
          ))}
        </svg>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 22: Lo que no modelamos ────────────────────────────
const S22 = ({step}) => (
  <SlideChrome num={22}>
    <GT size={36}>Fuera del foco de esta tesis</GT>
    <Col gap={20} style={{marginTop:30}}>
      {[
        {at:1,text:'Progresiones armónicas'},
        {at:2,text:'Conducción de voces'},
        {at:3,text:'Función tonal'},
        {at:4,text:'Temporalidad musical'},
      ].map(({at,text})=>(
        <Appear key={at} at={at} step={step}>
          <Row gap={16} align="center">
            <div style={{width:30,height:30,borderRadius:'50%',
              background:'#fee',border:`1.5px solid ${C.red}`,
              display:'flex',alignItems:'center',justifyContent:'center',
              fontSize:18,color:C.red,fontWeight:700}}>✗</div>
            <GT size={26} color={C.gray}>{text}</GT>
          </Row>
        </Appear>
      ))}
    </Col>
  </SlideChrome>
);

window.SLIDES_A = [
  {steps:0,slide:S1},{steps:1,slide:S2},{steps:3,slide:S3},{steps:3,slide:S4},
  {steps:3,slide:S5},{steps:2,slide:S6},{steps:1,slide:S7},{steps:2,slide:S8},
  {steps:4,slide:S9},{steps:3,slide:S10},{steps:1,slide:S11},{steps:3,slide:S12},
  {steps:2,slide:S13},{steps:1,slide:S14},{steps:2,slide:S15},{steps:1,slide:S16},
  {steps:1,slide:S17},{steps:3,slide:S18},{steps:2,slide:S19},{steps:2,slide:S20},
  {steps:2,slide:S21},{steps:4,slide:S22},
];
