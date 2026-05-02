// slides-b.jsx — Diapositivas 23–44
// Vacío + Objetivos + Metodología

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

// ─── Slide 23: Vacío en el estado del arte ─────────────────────
const S23 = ({step}) => (
  <SlideChrome num={23}>
    <GT size={36}>Vacío que ocupa esta tesis</GT>
    <div style={{height:'calc(100% - 90px)',display:'flex',alignItems:'center',justifyContent:'center'}}>
      <svg width={700} height={380} style={{overflow:'visible'}}>
        <defs>
          <clipPath id="clip1"><ellipse cx={210} cy={180} rx={180} ry={130}/></clipPath>
          <clipPath id="clip2"><ellipse cx={490} cy={180} rx={180} ry={130}/></clipPath>
        </defs>
        <Appear at={1} step={step} style={{display:'contents'}}>
          <ellipse cx={210} cy={180} rx={180} ry={130} fill={C.blue+'22'} stroke={C.blue} strokeWidth="1.5"/>
          <text x={120} y={170} textAnchor="middle" fontSize="15" fill={C.blue} fontFamily="Raleway,sans-serif" fontWeight="700">Taxonomía</text>
          <text x={120} y={190} textAnchor="middle" fontSize="12" fill={C.blue} fontFamily="Raleway,sans-serif">Forte, PC-sets</text>
        </Appear>
        <Appear at={2} step={step} style={{display:'contents'}}>
          <ellipse cx={490} cy={180} rx={180} ry={130} fill={C.dgreen+'22'} stroke={C.dgreen} strokeWidth="1.5"/>
          <text x={590} y={170} textAnchor="middle" fontSize="15" fill={C.dgreen} fontFamily="Raleway,sans-serif" fontWeight="700">Geometría</text>
          <text x={590} y={190} textAnchor="middle" fontSize="12" fill={C.dgreen} fontFamily="Raleway,sans-serif">Tymoczko</text>
        </Appear>
        <Appear at={3} step={step} style={{display:'contents'}}>
          <ellipse cx={350} cy={260} rx={150} ry={100} fill={C.orange+'33'} stroke={C.orange} strokeWidth="2"/>
          <text x={350} y={258} textAnchor="middle" fontSize="15" fill={C.orange} fontFamily="Raleway,sans-serif" fontWeight="700">Percepción</text>
          <text x={350} y={278} textAnchor="middle" fontSize="12" fill={C.orange} fontFamily="Raleway,sans-serif">huella rugosidad</text>
        </Appear>
        <Appear at={4} step={step} style={{display:'contents'}}>
          <circle cx={350} cy={215} r={18} fill={C.green}/>
          <text x={350} y={220} textAnchor="middle" fontSize="10" fill="white" fontFamily="Raleway,sans-serif" fontWeight="700">★</text>
          <text x={350} y={160} textAnchor="middle" fontSize="14" fill={C.green} fontFamily="Raleway,sans-serif" fontWeight="700">esta tesis</text>
        </Appear>
      </svg>
    </div>
  </SlideChrome>
);

// ─── Slide 24: Objetivo general ─────────────────────────────────
const S24 = ({step}) => (
  <SlideChrome num={24}>
    <div style={{height:'100%',display:'flex',flexDirection:'column',alignItems:'center',justifyContent:'center',gap:24}}>
      <Sub size={14} color={C.gold} style={{letterSpacing:'0.14em'}}>Objetivo general</Sub>
      <Appear at={1} step={step} style={{textAlign:'center',maxWidth:860}}>
        <GT size={32} style={{textAlign:'center'}}>
          Desarrollar un modelo computacional para la construcción y exploración
          de un espacio de representación y sustitución armónica.
        </GT>
      </Appear>
    </div>
  </SlideChrome>
);

// ─── Slide 25: Obj. específicos I ──────────────────────────────
const S25 = ({step}) => (
  <SlideChrome num={25}>
    <GT size={36}>Objetivos específicos</GT>
    <Row gap={80} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      {[
        {at:1,n:'01',text:'Modelar matemáticamente la generación y caracterización de acordes',color:C.green},
        {at:2,n:'02',text:'Construir una representación intervalar y perceptual del acorde',color:C.blue},
      ].map(({at,n,text,color})=>(
        <Appear key={n} at={at} step={step}>
          <Col gap={12} align="center" style={{width:300}}>
            <div style={{fontSize:56,fontWeight:800,color,fontFamily:"'Playfair Display',serif",lineHeight:1}}>{n}</div>
            <GT size={20} color={C.black} style={{textAlign:'center'}}>{text}</GT>
          </Col>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 26: Obj. específicos II ─────────────────────────────
const S26 = ({step}) => (
  <SlideChrome num={26}>
    <GT size={36}>Objetivos específicos</GT>
    <Row gap={80} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      {[
        {at:1,n:'03',text:'Implementar reducción dimensional para visualizar el espacio inducido',color:C.orange},
        {at:2,n:'04',text:'Evaluar cuantitativamente la calidad de la representación',color:C.dgreen},
        {at:3,n:'05',text:'Explorar una noción de sustitución armónica por proximidad métrica',color:C.green},
      ].map(({at,n,text,color})=>(
        <Appear key={n} at={at} step={step}>
          <Col gap={10} align="center" style={{width:220}}>
            <div style={{fontSize:48,fontWeight:800,color,fontFamily:"'Playfair Display',serif",lineHeight:1}}>{n}</div>
            <GT size={18} color={C.black} style={{textAlign:'center'}}>{text}</GT>
          </Col>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 27: Decisión metodológica ───────────────────────────
const S27 = ({step}) => (
  <SlideChrome num={27}>
    <GT size={36}>Unidad de análisis</GT>
    <Row gap={80} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      <Appear at={1} step={step}>
        <Col gap={16} align="center">
          <div style={{fontSize:60,color:'#ddd',textDecoration:'line-through',
            fontFamily:"'Playfair Display',serif",fontWeight:800}}>C → F → G → C</div>
          <div style={{width:32,height:32,borderRadius:'50%',background:'#fee',
            border:`1.5px solid ${C.red}`,display:'flex',alignItems:'center',justifyContent:'center',
            fontSize:22,color:C.red}}>✗</div>
          <Label size={16} color={C.gray}>progresión</Label>
        </Col>
      </Appear>
      <Appear at={2} step={step}>
        <Col gap={16} align="center">
          <PianoKeyboard startNote={48} endNote={55}
            dots={{48:{color:C.green},52:{color:C.green},55:{color:C.green}}}
            W={38} showMidi={false}/>
          <div style={{width:32,height:32,borderRadius:'50%',background:C.green+'22',
            border:`1.5px solid ${C.green}`,display:'flex',alignItems:'center',justifyContent:'center',
            fontSize:22,color:C.green}}>✓</div>
          <Label size={16} color={C.green} style={{fontWeight:700}}>acorde aislado</Label>
        </Col>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 28: Sistema MIDI ─────────────────────────────────────
const S28 = ({step}) => (
  <SlideChrome num={28}>
    <GT size={36}>Sistema de referencia</GT>
    <Row gap={50} align="center" style={{height:'calc(100% - 90px)'}}>
      <Appear at={1} step={step}>
        <PianoKeyboard startNote={36} endNote={60} separatorAt={48} W={30}
          onPlay={n=>AudioEngine.playNote(n,1.2)}/>
      </Appear>
      <Col gap={20}>
        <Appear at={2} step={step}>
          <Eq size={17}>f(n) = 440 · 2^((n–69)/12)</Eq>
        </Appear>
        <Appear at={2} step={step}>
          <Label size={16} color={C.gray}>A4 = 440 Hz, 12-TET</Label>
        </Appear>
        <Appear at={3} step={step}>
          <Row gap={12}>
            {[[48,'Do₃',C.green],[52,'Mi₃',C.blue],[55,'Sol₃',C.dgreen]].map(([n,name,col])=>(
              <Col key={n} gap={4} align="center">
                <SoundBtn onClick={()=>AudioEngine.playNote(n)} color={col} size={32}/>
                <Label size={12} color={col} style={{fontWeight:700}}>{name}</Label>
                <Label size={11} color={C.gray}>midi {n}</Label>
              </Col>
            ))}
          </Row>
        </Appear>
      </Col>
    </Row>
  </SlideChrome>
);

// ─── Slide 29: Definición del acorde ───────────────────────────
const S29 = ({step}) => (
  <SlideChrome num={29}>
    <GT size={36}>Definición formal</GT>
    <Col gap={28} style={{marginTop:24,alignItems:'flex-start'}}>
      <Appear at={1} step={step}>
        <Eq size={20}>n = (n₁, n₂, …, nₘ)  con  n₁ &lt; n₂ &lt; ··· &lt; nₘ</Eq>
      </Appear>
      <Appear at={2} step={step}>
        <PianoKeyboard startNote={48} endNote={60}
          dots={{48:{color:C.red},52:{color:C.blue},55:{color:C.dgreen}}}
          noteLabels={{48:{text:'n₁',color:C.red},52:{text:'n₂',color:C.blue},55:{text:'n₃',color:C.dgreen}}}
          W={38} onPlay={n=>AudioEngine.playNote(n)}/>
      </Appear>
      <Appear at={3} step={step}>
        <Eq size={18}>C_M = (48, 52, 55)</Eq>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 30: Registro importa ────────────────────────────────
const S30 = ({step}) => (
  <SlideChrome num={30}>
    <GT size={36}>El registro importa</GT>
    <Row gap={60} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      {[
        {at:1,start:36,notes:[36,40,43],label:'C₂ mayor',sub:'rugosidad alta',color:C.red},
        {at:2,start:48,notes:[48,52,55],label:'C₃ mayor',sub:'rugosidad media',color:C.orange},
        {at:3,start:60,notes:[60,64,67],label:'C₄ mayor',sub:'rugosidad baja',color:C.dgreen},
      ].map(({at,start,notes,label,sub,color})=>(
        <Appear key={at} at={at} step={step}>
          <Col gap={10} align="center">
            <PianoKeyboard startNote={start-1} endNote={notes[2]+2}
              dots={Object.fromEntries(notes.map(n=>[n,{color}]))}
              W={32} showMidi={false}
              onPlay={()=>AudioEngine.playChord(notes)}/>
            <SoundBtn onClick={()=>AudioEngine.playChord(notes)} color={color} size={30}/>
            <GT size={18} color={color}>{label}</GT>
            <Label size={14} color={C.gray}>{sub}</Label>
          </Col>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 31: Estructura interválica ──────────────────────────
const S31 = ({step}) => (
  <SlideChrome num={31}>
    <GT size={36}>Estructura interválica</GT>
    <Col gap={20} style={{marginTop:20,alignItems:'center'}}>
      <Appear at={1} step={step}>
        <PianoKeyboard startNote={48} endNote={55}
          dots={{48:{color:C.red},52:{color:C.blue},55:{color:C.dgreen}}}
          W={42}/>
      </Appear>
      <Appear at={2} step={step}>
        <svg width={500} height={80} style={{overflow:'visible'}}>
          <defs>
            <marker id="ai" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
              <path d="M0,0 L0,7 L7,3.5 Z" fill={C.black}/>
            </marker>
          </defs>
          {[
            {x1:50,y1:20,x2:250,y2:20,label:'4 st',col:C.blue},
            {x1:250,y1:50,x2:380,y2:50,label:'3 st',col:C.dgreen},
            {x1:50,y1:65,x2:380,y2:65,label:'7 st',col:C.gray},
          ].map(({x1,y1,x2,y2,label,col},i)=>(
            <g key={i}>
              <path d={`M${x1},${y1} Q${(x1+x2)/2},${y1-20} ${x2},${y1}`} fill="none"
                stroke={col} strokeWidth="2" markerEnd="url(#ai)"/>
              <text x={(x1+x2)/2} y={y1-12} textAnchor="middle" fontSize="13" fill={col}
                fontWeight="700" fontFamily="Raleway,sans-serif">{label}</text>
            </g>
          ))}
        </svg>
      </Appear>
      <Appear at={3} step={step}>
        <Eq size={17}>ic_k(n) = #{'{'}(i,j): i&lt;j, (nⱼ−nᵢ) ≡ k (mod 12){'}'}</Eq>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 32: 12 bins, no 6 ────────────────────────────────────
const S32 = ({step}) => (
  <SlideChrome num={32}>
    <GT size={36}>12 bins, no 6</GT>
    <Row gap={60} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      <Appear at={1} step={step}>
        <Col gap={14} align="center">
          <Sub size={12} color={C.gray}>Con 6 bins (Forte)</Sub>
          <svg width={220} height={100}>
            {Array.from({length:6},(_,i)=>(
              <g key={i}>
                <rect x={i*36+2} y={20} width={32} height={60} fill={C.gray} opacity="0.4" rx="1"/>
                <text x={i*36+18} y={98} textAnchor="middle" fontSize="10" fill={C.gray} fontFamily="Raleway,sans-serif">{i+1}</text>
              </g>
            ))}
          </svg>
          <Label size={14} color={C.gray}>ic1 = ic11 fusionados</Label>
        </Col>
      </Appear>
      <Appear at={2} step={step}>
        <Col gap={14} align="center">
          <Sub size={12} color={C.green}>Con 12 bins (esta tesis)</Sub>
          <svg width={420} height={100}>
            {Array.from({length:12},(_,i)=>(
              <g key={i}>
                <rect x={i*34+2} y={i<6?10:30} width={30} height={i<6?70:50}
                  fill={i<6?C.green:C.blue} opacity="0.5" rx="1"/>
                <text x={i*34+17} y={98} textAnchor="middle" fontSize="10"
                  fill={i<6?C.green:C.blue} fontFamily="Raleway,sans-serif">{i+1}</text>
              </g>
            ))}
          </svg>
          <Label size={14} color={C.green}>ic1 ≠ ic11 — dirección preservada</Label>
        </Col>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 33: Exploración combinatoria ────────────────────────
const S33 = ({step}) => (
  <SlideChrome num={33}>
    <GT size={36}>Exploración combinatoria</GT>
    <Row gap={50} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      <Col gap={12}>
        {[
          {at:1,label:'alfabeto',val:'12 clases cromáticas'},
          {at:2,label:'cardinalidad',val:'2, 3, 4, 5 notas'},
          {at:3,label:'rango',val:'1–3 octavas'},
          {at:4,label:'anclaje',val:'nota grave fija'},
        ].map(({at,label,val})=>(
          <Appear key={at} at={at} step={step}>
            <Row gap={20} align="center">
              <Sub size={12} color={C.gold} style={{width:120,letterSpacing:'0.08em'}}>{label}</Sub>
              <Label size={17} color={C.black}>{val}</Label>
            </Row>
          </Appear>
        ))}
      </Col>
      <Appear at={4} step={step}>
        <ArrowH color={C.gold} length={50}/>
      </Appear>
      <Appear at={4} step={step}>
        <Col gap={8} align="center">
          <div style={{fontSize:52,color:C.green,fontFamily:"'Playfair Display',serif",fontWeight:800}}>U</div>
          <Label size={14} color={C.gray}>universo controlado</Label>
        </Col>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 34: Anclaje en Do ────────────────────────────────────
const S34 = ({step}) => (
  <SlideChrome num={34}>
    <GT size={36}>Anclaje en Do</GT>
    <Col gap={24} style={{marginTop:24,alignItems:'center'}}>
      <Appear at={1} step={step}>
        <GT size={20} italic color={C.black} style={{textAlign:'center',maxWidth:700}}>
          Anclar el acorde no significa perder generalidad musical,<br/>
          sino ganar <span style={{color:C.green}}>control experimental</span>.
        </GT>
      </Appear>
      <Appear at={2} step={step}>
        <Row gap={20} align="center">
          {['D_M','E_M','F_M','G_M'].map((ch,i)=>(
            <React.Fragment key={ch}>
              <div style={{fontSize:22,color:C.gray,fontFamily:"'JetBrains Mono',monospace"}}>{ch}</div>
              {i<3&&<ArrowH color={C.gray} length={40}/>}
            </React.Fragment>
          ))}
          <ArrowH color={C.gold} length={40}/>
          <div style={{fontSize:22,color:C.green,fontFamily:"'JetBrains Mono',monospace",fontWeight:700}}>C_M</div>
        </Row>
      </Appear>
      <Appear at={2} step={step}>
        <Sub size={12} color={C.gray} style={{letterSpacing:'0.1em'}}>transposición global → bajo = C fijo</Sub>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 35: Filtros y parámetros ────────────────────────────
const S35 = ({step}) => (
  <SlideChrome num={35}>
    <GT size={36}>Parámetros del espacio</GT>
    <Row gap={48} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      {[
        {at:1,title:'Cardinalidad',opts:['|n|=2','|n|=3','|n|=4','|n|=5']},
        {at:2,title:'Rango MIDI',opts:['[48, 60]','[36, 72]','[24, 84]']},
        {at:3,title:'Distancia mínima',opts:['Δ ≥ 1 st','Δ ≥ 2 st']},
      ].map(({at,title,opts})=>(
        <Appear key={at} at={at} step={step}>
          <Col gap={10}>
            <Sub size={12} color={C.gold} style={{letterSpacing:'0.1em'}}>{title}</Sub>
            {opts.map(o=>(
              <div key={o} style={{border:`1px solid #ddd`,padding:'6px 16px',
                fontFamily:"'JetBrains Mono',monospace",fontSize:15,color:C.black}}>{o}</div>
            ))}
          </Col>
        </Appear>
      ))}
    </Row>
  </SlideChrome>
);

// ─── Slide 36: Modelo de rugosidad ─────────────────────────────
const S36 = ({step}) => {
  const W=440,H=180;
  const curve = Array.from({length:100},(_,i)=>{
    const x=i/99; const r=x*Math.exp(1-x/0.25)*0.25;
    return [i*(W/99), H-r*H*0.92];
  });
  return(
    <SlideChrome num={36}>
      <GT size={36}>Modelo de rugosidad — Sethares</GT>
      <Row gap={50} align="center" style={{height:'calc(100% - 90px)'}}>
        <Appear at={1} step={step}>
          <svg width={W} height={H+40} style={{overflow:'visible'}}>
            <line x1="0" y1={H} x2={W} y2={H} stroke="#bbb" strokeWidth="1.5"/>
            <line x1="0" y1="0" x2="0" y2={H} stroke="#bbb" strokeWidth="1.5"/>
            <polyline points={curve.map(([x,y])=>`${x},${y}`).join(' ')}
              fill="none" stroke={C.orange} strokeWidth="3"/>
            <text x={W/2} y={H+28} textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif">Δf / banda crítica</text>
            <text x={-12} y={H/2} textAnchor="middle" fontSize="12" fill={C.gray} fontFamily="Raleway,sans-serif" transform={`rotate(-90,-12,${H/2})`}>d(fa,fb)</text>
          </svg>
        </Appear>
        <Appear at={2} step={step}>
          <Col gap={16}>
            <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:14,
              border:'1px dashed #555',padding:'12px 18px',lineHeight:1.8}}>
              d(fa,fb,Aa,Ab) =<br/>
              &nbsp;Aa·Ab·(e^(−b₁·s·Δf) − e^(−b₂·s·Δf))
            </div>
            <Label size={14} color={C.gray}>b₁=3.5 &nbsp; b₂=5.75 &nbsp; s ajusta banda crítica</Label>
          </Col>
        </Appear>
      </Row>
    </SlideChrome>
  );
};

// ─── Slide 37: Tonos complejos ──────────────────────────────────
const S37 = ({step}) => (
  <SlideChrome num={37}>
    <GT size={36}>Tonos complejos — serie armónica</GT>
    <Col gap={24} style={{marginTop:24,alignItems:'center'}}>
      <Appear at={1} step={step}>
        <Row gap={6} align="flex-end" justify="center">
          {Array.from({length:6},(_,i)=>{
            const h=Math.round(120*Math.pow(0.88,i));
            return(
              <Col key={i} gap={6} align="center">
                <Label size={12} color={C.blue} style={{fontWeight:700}}>{Math.round(Math.pow(0.88,i)*100)}%</Label>
                <div style={{width:40,height:h,background:C.blue,opacity:0.6+(i*0.06),borderRadius:'2px 2px 0 0'}}/>
                <Label size={12} color={C.gray}>k={i+1}</Label>
                <Label size={11} color={C.gray} style={{fontFamily:"'JetBrains Mono',monospace"}}>f·{i+1}</Label>
              </Col>
            );
          })}
        </Row>
      </Appear>
      <Appear at={2} step={step}>
        <Row gap={30}>
          <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:15,
            border:'1px dashed #555',padding:'10px 18px',lineHeight:1.7}}>
            Ak = δ^(k−1) · A₀<br/>
            δ = 0.88,  H = 6
          </div>
          <SoundBtn onClick={()=>AudioEngine.playNote(48,2.5)} size={44} color={C.green}/>
        </Row>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 38: Del acorde al perfil rugoso (pipeline central) ──
const S38 = ({step}) => (
  <SlideChrome num={38}>
    <GT size={36}>Del acorde al perfil de rugosidad</GT>
    <Col gap={12} style={{marginTop:16,alignItems:'center'}}>
      {/* Step 1: acorde */}
      <Appear at={1} step={step}>
        <PianoKeyboard startNote={48} endNote={55}
          dots={{48:{color:C.red},52:{color:C.blue},55:{color:C.dgreen}}}
          W={38} showMidi={false}
          onPlay={()=>AudioEngine.playChord([48,52,55])}/>
      </Appear>
      <Appear at={1} step={step}><ArrowH color={C.gold} length={40}/></Appear>

      {/* Step 2: pares */}
      <Appear at={2} step={step}>
        <Row gap={12} align="center">
          {[['(48,52)','4st',C.blue],['(48,55)','7st',C.gray],['(52,55)','3st',C.dgreen]].map(([pair,st,col])=>(
            <div key={pair} style={{fontFamily:"'JetBrains Mono',monospace",fontSize:14,
              padding:'6px 12px',border:`1.5px solid ${col}`,color:col}}>{pair}<br/><span style={{fontSize:11,color:C.gray}}>{st}</span></div>
          ))}
          <Label size={16} color={C.gray} style={{marginLeft:8}}>→ R(nᵢ,nⱼ)</Label>
        </Row>
      </Appear>
      <Appear at={2} step={step}><ArrowH color={C.gold} length={40}/></Appear>

      {/* Step 3: bins */}
      <Appear at={3} step={step}>
        <Row gap={3} align="flex-end">
          {Array.from({length:12},(_,k)=>{
            const vals=[0,0,0,0.7,0,0,0,0.4,0,0,0,0]; // C major example
            const h=Math.round(vals[k]*80)+4;
            const colors=[C.gray,C.gray,C.gray,C.dgreen,C.gray,C.gray,C.gray,C.blue,C.gray,C.gray,C.gray,C.gray];
            return(
              <Col key={k} gap={4} align="center">
                <div style={{width:28,height:h,background:colors[k],borderRadius:'2px 2px 0 0',opacity:0.7}}/>
                <Label size={9} color={C.gray} style={{fontFamily:"'JetBrains Mono',monospace"}}>{k+1}</Label>
              </Col>
            );
          })}
        </Row>
      </Appear>
      <Appear at={3} step={step}><ArrowH color={C.gold} length={40}/></Appear>

      {/* Step 4: vector */}
      <Appear at={4} step={step}>
        <BinaryVector
          values={[0,0,0,'R₄',0,0,0,'R₇',0,0,0,0]}
          colors={{3:C.dgreen,7:C.blue}}
          labels={['1','2','3','4','5','6','7','8','9','10','11','12']}
          cellW={44}/>
      </Appear>
      <Appear at={4} step={step}>
        <Sub size={12} color={C.green} style={{letterSpacing:'0.1em',marginTop:4}}>Φ_raw ∈ ℝ¹²</Sub>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 39: Rugosidad total vs perfil 12D ────────────────────
const S39 = ({step}) => {
  const vals=[0.05,0.08,0.12,0.72,0.04,0.09,0.06,0.41,0.03,0.05,0.08,0.02];
  const total=vals.reduce((a,b)=>a+b,0).toFixed(2);
  return(
    <SlideChrome num={39}>
      <GT size={36}>Rugosidad total vs. perfil 12D</GT>
      <Row gap={80} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
        <Appear at={1} step={step}>
          <Col gap={14} align="center">
            <Sub size={12} color={C.gray}>Escalar (baseline)</Sub>
            <div style={{fontSize:72,fontWeight:800,color:C.gray,fontFamily:"'Playfair Display',serif"}}>{total}</div>
            <Label size={14} color={C.gray}>R_total = ‖Φ_raw‖₁</Label>
            <Label size={13} color:C.orange style={{color:C.orange,fontStyle:'italic'}}>pierde distribución interna</Label>
          </Col>
        </Appear>
        <Appear at={2} step={step}>
          <Col gap={14} align="center">
            <Sub size={12} color={C.green}>Perfil 12D (esta tesis)</Sub>
            <RoughnessProfile values={vals} barColor={C.green} width={340} height={110}/>
            <Label size={14} color={C.green}>Φ_raw ∈ ℝ¹²</Label>
            <Label size={13} style={{color:C.green,fontStyle:'italic'}}>preserva dónde está la rugosidad</Label>
          </Col>
        </Appear>
      </Row>
    </SlideChrome>
  );
};

// ─── Slide 40: Bin 12 y octavas ────────────────────────────────
const S40 = ({step}) => (
  <SlideChrome num={40}>
    <GT size={36}>El bin 12 — octavas</GT>
    <Col gap={24} style={{marginTop:24,alignItems:'center'}}>
      <Appear at={1} step={step}>
        <Row gap={2} align="flex-end">
          {Array.from({length:12},(_,i)=>(
            <Col key={i} gap={4} align="center">
              <div style={{width:44,height:i===11?90:50,
                background:i===11?C.gold:C.blue,
                borderRadius:'2px 2px 0 0',opacity:i===11?1:0.55}}/>
              <Label size={10} color={i===11?C.gold:C.gray}
                style={{fontFamily:"'JetBrains Mono',monospace"}}>{i+1}</Label>
            </Col>
          ))}
        </Row>
      </Appear>
      <Appear at={2} step={step}>
        <Row gap={40} align="center">
          <Col gap={6} align="center">
            <PianoKeyboard startNote={48} endNote={60}
              dots={{48:{color:C.gold},60:{color:C.gold}}}
              W={28} showMidi={false} showSolfege={false}/>
            <Label size={13} color={C.gold} style={{fontWeight:700}}>Δ = 12 st (octava)</Label>
          </Col>
          <Label size={16} color={C.gray}>→</Label>
          <Col gap={6} align="center">
            <Label size={18} color={C.gold} style={{fontWeight:700}}>bin 12</Label>
            <Label size={13} color={C.gray}>alineación espectral,<br/>rugosidad ≈ 0</Label>
          </Col>
        </Row>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 41: Normalización ────────────────────────────────────
const S41 = ({step}) => {
  const v3=[0.05,0.08,0.12,0.72,0.04,0.09,0.06,0.41,0.03,0.05,0.08,0.02];
  const v4=v3.map(x=>x*1.6);
  const norm=(v,c)=>v.map(x=>x/Math.pow(c,0.75));
  const n3=norm(v3,3),n4=norm(v4,4);
  return(
    <SlideChrome num={41}>
      <GT size={36}>Normalización por cardinalidad</GT>
      <Row gap={60} align="flex-start" justify="center" style={{marginTop:20}}>
        <Appear at={1} step={step}>
          <Col gap={12} align="center">
            <Sub size={11} color={C.gray}>Sin normalizar — tríada</Sub>
            <RoughnessProfile values={v3} barColor={C.blue} width={240} height={80}/>
            <Sub size={11} color={C.gray}>Sin normalizar — tétrada</Sub>
            <RoughnessProfile values={v4} barColor={C.red} width={240} height={80}/>
          </Col>
        </Appear>
        <Appear at={2} step={step}>
          <ArrowH color={C.gold} length={50} label="α=0.75"/>
        </Appear>
        <Appear at={2} step={step}>
          <Col gap={12} align="center">
            <Sub size={11} color={C.green}>Normalizado — tríada</Sub>
            <RoughnessProfile values={n3} barColor={C.green} width={240} height={80}/>
            <Sub size={11} color={C.green}>Normalizado — tétrada</Sub>
            <RoughnessProfile values={n4} barColor={C.green} width={240} height={80}/>
          </Col>
        </Appear>
      </Row>
    </SlideChrome>
  );
};

// ─── Slide 42: Por qué α=0.75 ──────────────────────────────────
const S42 = ({step}) => (
  <SlideChrome num={42}>
    <GT size={36}>Elección de α = 0.75</GT>
    <Col gap={24} style={{marginTop:24,alignItems:'center'}}>
      <Appear at={1} step={step}>
        <Row gap={0}>
          {[
            {alpha:'0',desc:'elimina magnitud',ok:false},
            {alpha:'0.75',desc:'equilibrio',ok:true},
            {alpha:'1',desc:'no corrige cardinalidad',ok:false},
          ].map(({alpha,desc,ok})=>(
            <Col key={alpha} gap={8} align="center" style={{
              flex:1,padding:'20px 16px',
              border:`2px solid ${ok?C.green:'#ddd'}`,
              background:ok?C.green+'11':'transparent',
              margin:'0 4px',
            }}>
              <div style={{fontFamily:"'Playfair Display',serif",fontSize:36,fontWeight:800,
                color:ok?C.green:C.gray}}>α={alpha}</div>
              <Label size={14} color={ok?C.green:C.gray}>{desc}</Label>
              {ok&&<div style={{fontSize:20,color:C.green}}>✓</div>}
            </Col>
          ))}
        </Row>
      </Appear>
      <Appear at={2} step={step}>
        <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:16,
          border:'1px dashed #555',padding:'10px 20px'}}>
          Φ_α,k = Φ_raw,k / max(mₖ, 1)^α
        </div>
      </Appear>
    </Col>
  </SlideChrome>
);

// ─── Slide 43: Distancias ───────────────────────────────────────
const S43 = ({step}) => (
  <SlideChrome num={43}>
    <GT size={36}>Distancias en ℝ¹²</GT>
    <Row gap={60} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
      <Appear at={1} step={step}>
        <Col gap={16} align="center">
          <Sub size={12} color={C.green}>Euclidiana (principal)</Sub>
          <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:16,
            border:'1px dashed '+C.green,padding:'12px 20px'}}>
            d(x,y) = ‖Φ(x) − Φ(y)‖₂
          </div>
        </Col>
      </Appear>
      <Appear at={2} step={step}>
        <Col gap={16} align="center">
          <Sub size={12} color={C.blue}>Coseno (complementaria)</Sub>
          <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:16,
            border:'1px dashed '+C.blue,padding:'12px 20px'}}>
            sim(x,y) = Φ(x)·Φ(y) / (‖x‖‖y‖)
          </div>
        </Col>
      </Appear>
    </Row>
  </SlideChrome>
);

// ─── Slide 44: Pipeline metodológico ───────────────────────────
const S44 = ({step}) => {
  const boxes=[
    {at:1,label:'Generar\nacorados',color:C.green},
    {at:2,label:'Vectorizar\n(Φ ∈ ℝ¹²)',color:C.blue},
    {at:3,label:'Medir\ndistancias',color:C.orange},
    {at:4,label:'Proyectar\n(MDS/UMAP)',color:C.dgreen},
    {at:5,label:'Explorar\nsustituciones',color:C.green},
  ];
  return(
    <SlideChrome num={44}>
      <GT size={36}>Pipeline metodológico</GT>
      <Row gap={0} align="center" justify="center" style={{height:'calc(100% - 90px)'}}>
        {boxes.map(({at,label,color},i)=>(
          <React.Fragment key={at}>
            <Appear at={at} step={step}>
              <div style={{border:`2px solid ${color}`,padding:'18px 16px',
                textAlign:'center',width:150,background:color+'11'}}>
                <GT size={17} color={color} style={{whiteSpace:'pre-line',textAlign:'center'}}>{label}</GT>
              </div>
            </Appear>
            {i<boxes.length-1&&<Appear at={at} step={step}><ArrowH color={C.gold} length={36}/></Appear>}
          </React.Fragment>
        ))}
      </Row>
    </SlideChrome>
  );
};

window.SLIDES_B = [
  {steps:4,slide:S23},{steps:1,slide:S24},{steps:2,slide:S25},{steps:3,slide:S26},
  {steps:2,slide:S27},{steps:3,slide:S28},{steps:3,slide:S29},{steps:3,slide:S30},
  {steps:3,slide:S31},{steps:2,slide:S32},{steps:4,slide:S33},{steps:2,slide:S34},
  {steps:3,slide:S35},{steps:2,slide:S36},{steps:2,slide:S37},{steps:4,slide:S38},
  {steps:2,slide:S39},{steps:2,slide:S40},{steps:2,slide:S41},{steps:2,slide:S42},
  {steps:2,slide:S43},{steps:5,slide:S44},
];
