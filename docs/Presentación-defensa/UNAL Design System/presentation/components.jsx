// components.jsx — Shared components for UNAL Thesis Defense
// Exports to window: SlideChrome, PianoKeyboard, ChromaticCircle,
//   BinaryVector, RoughnessProfile, Appear, SoundBtn, Arrow, Col, Row

const { useState, useEffect, useRef, useCallback } = React;

// ─── UNAL Colors ────────────────────────────────────────────────────────────
const C = {
  green:'#1B7A3E', greenL:'#2DC56E', gold:'#E8A020', orange:'#E8610A',
  beige:'#F0EBE0', white:'#FFFFFF', black:'#1A1A1A',
  blue:'#1E6BB8', red:'#C0392B', dgreen:'#27AE60', gray:'#888',
};

// ─── Layout helpers ─────────────────────────────────────────────────────────
const Row = ({gap=16,align='center',justify='flex-start',style={},children}) => (
  <div style={{display:'flex',flexDirection:'row',gap,alignItems:align,justifyContent:justify,...style}}>{children}</div>
);
const Col = ({gap=12,align='flex-start',justify='flex-start',style={},children}) => (
  <div style={{display:'flex',flexDirection:'column',gap,alignItems:align,justifyContent:justify,...style}}>{children}</div>
);

// ─── Appear (step-gated fade-in) ────────────────────────────────────────────
const Appear = ({at=1,step=0,children,style={}}) => (
  <div style={{
    opacity: step>=at ? 1 : 0,
    transform: `translateY(${step>=at ? 0 : 10}px)`,
    transition:'opacity 0.38s ease, transform 0.38s ease',
    ...style,
  }}>{children}</div>
);

// ─── Sound button ────────────────────────────────────────────────────────────
const SoundBtn = ({onClick,label='▶',color=C.green,size=36}) => (
  <button onClick={onClick} style={{
    width:size,height:size,borderRadius:'50%',
    background:'transparent',border:`1.5px solid ${color}`,
    color,fontSize:size*0.38,cursor:'pointer',
    display:'flex',alignItems:'center',justifyContent:'center',
    transition:'background 0.15s',flexShrink:0,
  }}
    onMouseEnter={e=>e.currentTarget.style.background=color+'22'}
    onMouseLeave={e=>e.currentTarget.style.background='transparent'}
  >{label}</button>
);

// ─── UNAL Slide Chrome ───────────────────────────────────────────────────────
const SlideChrome = ({num, dept='Facultad de Ciencias · Matemáticas Aplicadas', dark=false, children}) => (
  <div style={{
    width:1280, height:720, position:'relative', overflow:'hidden',
    background: dark ? C.green : C.white,
    fontFamily:"'Playfair Display','Georgia',serif",
  }}>
    {/* top-left page number */}
    {!dark && <div style={{position:'absolute',top:10,left:16,fontSize:11,
      fontFamily:"'Raleway',sans-serif",color:C.gray}}>{num}</div>}
    {/* gold accent top-right */}
    <div style={{position:'absolute',top:14,right:0,width:'56%',height:2,background:C.gold}}/>
    {/* content */}
    <div style={{position:'absolute',inset:'0 0 44px 0',padding:'48px 72px 0'}}>{children}</div>
    {/* bottom green lines */}
    <div style={{position:'absolute',bottom:50,left:24,width:52,height:2,background:C.greenL}}/>
    <div style={{position:'absolute',bottom:44,left:24,width:30,height:2,background:C.greenL}}/>
    {/* footer */}
    {!dark && <div style={{
      position:'absolute',bottom:0,left:0,right:0,height:40,
      background:C.beige,display:'flex',alignItems:'center',
      justifyContent:'space-between',padding:'0 16px',
    }}>
      <div style={{fontFamily:"'Raleway',sans-serif",fontSize:9,color:C.green,fontStyle:'italic',lineHeight:1.6}}>
        {dept.split('·').map((d,i)=><div key={i}>{d.trim()}</div>)}
      </div>
      <svg width="28" height="28" viewBox="0 0 52 52" fill="none">
        <circle cx="26" cy="26" r="22" stroke={C.green} strokeWidth="1.5" opacity="0.7"/>
        <rect x="19" y="14" width="14" height="16" stroke={C.green} strokeWidth="1.2" opacity="0.7"/>
        <path d="M19 24 Q26 30 33 24" stroke={C.green} strokeWidth="1" opacity="0.6" fill="none"/>
        <text x="26" y="22" textAnchor="middle" fontSize="7" fill={C.green} opacity="0.8" fontFamily="serif">π</text>
      </svg>
    </div>}
  </div>
);

// ─── Piano Keyboard ──────────────────────────────────────────────────────────
const IS_BLACK = n => [1,3,6,8,10].includes(n%12);
const SOL_MAP = {0:'Do',2:'Re',4:'Mi',5:'Fa',7:'Sol',9:'La',11:'Si'};
const SOL_BLACK = {1:'Do#',3:'Re#',6:'Fa#',8:'Sol#',10:'La#'};

const PianoKeyboard = ({
  startNote=36, endNote=60,
  dots={},      // {midi: {color}}
  noteLabels={}, // {midi: {text, color}} — below keyboard
  spans=[],     // [{from,to,color,label}]
  separatorAt=null,
  W=30,         // white key width
  showMidi=true,
  showSolfege=true,
  onPlay=null,
}) => {
  const WH=90, BH=56, BW=Math.round(W*0.62);
  // Build layout
  const layout={};
  let wi=0;
  for(let n=startNote;n<=endNote;n++){
    if(!IS_BLACK(n)){
      layout[n]={isB:false, x:wi*W, cx:wi*W+W/2, wi};
      wi++;
    } else {
      const cx=(wi-1)*W + W*0.68;
      layout[n]={isB:true, x:cx-BW/2, cx};
    }
  }
  const totalW=wi*W;

  const whites=Object.keys(layout).map(Number).filter(n=>!IS_BLACK(n)).sort((a,b)=>a-b);
  const blacks=Object.keys(layout).map(Number).filter(n=>IS_BLACK(n)).sort((a,b)=>a-b);
  const sepX=separatorAt!=null&&layout[separatorAt]?layout[separatorAt].x:null;

  return (
    <div style={{display:'inline-block',position:'relative',userSelect:'none'}}>
      <div style={{position:'relative',width:totalW,height:WH}}>
        {whites.map(n=>{
          const l=layout[n],dot=dots[n],pc=n%12;
          return(
            <div key={n} onClick={()=>onPlay&&onPlay(n)} style={{
              position:'absolute',left:l.x,top:0,width:W-1,height:WH,
              background:'#fff',border:'1px solid #333',borderTop:'2px solid #222',
              boxSizing:'border-box',display:'flex',flexDirection:'column',alignItems:'center',
              cursor:onPlay?'pointer':'default',zIndex:1,
            }}>
              {showMidi&&<div style={{fontSize:7.5,color:'#555',marginTop:2,lineHeight:1,fontFamily:"'JetBrains Mono',monospace"}}>{n}</div>}
              <div style={{flex:1}}/>
              {dot&&<div style={{width:13,height:13,borderRadius:'50%',background:dot.color||C.red,marginBottom:2,flexShrink:0}}/>}
              {showSolfege&&<div style={{fontSize:8.5,fontWeight:700,color:'#222',marginBottom:3,fontFamily:"'Raleway',sans-serif"}}>{SOL_MAP[pc]||''}</div>}
            </div>
          );
        })}
        {blacks.map(n=>{
          const l=layout[n],dot=dots[n];
          return(
            <div key={n} onClick={()=>onPlay&&onPlay(n)} style={{
              position:'absolute',left:l.x,top:0,width:BW,height:BH,
              background:'#111',borderRadius:'0 0 4px 4px',zIndex:2,
              display:'flex',flexDirection:'column',alignItems:'center',
              cursor:onPlay?'pointer':'default',
            }}>
              {showMidi&&<div style={{fontSize:7,color:'#aaa',marginTop:1.5,lineHeight:1,fontFamily:"'JetBrains Mono',monospace"}}>{n}</div>}
              <div style={{flex:1}}/>
              {dot&&<div style={{width:10,height:10,borderRadius:'50%',background:dot.color||C.red,marginBottom:4,flexShrink:0}}/>}
            </div>
          );
        })}
        {sepX!=null&&<div style={{position:'absolute',left:sepX,top:0,width:2,height:WH,background:'#222',zIndex:3}}/>}
      </div>

      {/* noteLabels row */}
      {Object.keys(noteLabels).length>0&&(
        <div style={{position:'relative',width:totalW,height:22}}>
          {Object.entries(noteLabels).map(([nk,lbl])=>{
            const l=layout[Number(nk)];
            if(!l)return null;
            return(
              <div key={nk} style={{
                position:'absolute',left:l.cx-14,top:2,width:28,
                textAlign:'center',fontSize:14,fontWeight:700,
                color:lbl.color||C.red,fontFamily:"'Raleway',sans-serif",
              }}>{lbl.text}</div>
            );
          })}
        </div>
      )}

      {/* spans */}
      {spans.length>0&&(
        <svg width={totalW} height={36} style={{display:'block',overflow:'visible'}}>
          {spans.map((sp,i)=>{
            const l1=layout[sp.from],l2=layout[sp.to];
            if(!l1||!l2)return null;
            const x1=l1.cx,x2=l2.cx,y=20,col=sp.color||C.dgreen;
            const dx=10;
            return(
              <g key={i}>
                <polygon points={`${x1},${y} ${x1+dx*0.6},${y-5} ${x1+dx},${y} ${x1+dx*0.6},${y+5}`} fill="none" stroke={col} strokeWidth="1.5"/>
                <line x1={x1+dx} y1={y} x2={x2-dx} y2={y} stroke={col} strokeWidth="1.5"/>
                <polygon points={`${x2},${y} ${x2-dx*0.6},${y-5} ${x2-dx},${y} ${x2-dx*0.6},${y+5}`} fill="none" stroke={col} strokeWidth="1.5"/>
                {sp.label&&<text x={(x1+x2)/2} y={y-7} textAnchor="middle" fontSize="13" fill={col} fontWeight="700" fontFamily="Raleway,sans-serif">{sp.label}</text>}
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
};

// ─── Chromatic Circle ────────────────────────────────────────────────────────
const ChromaticCircle = ({size=190,highlighted={},dots=[],arcs=[]}) => {
  const cx=size/2,cy=size/2;
  const outerR=size*0.42,innerR=size*0.22;
  const midR=(outerR+innerR)/2;
  const labelR=outerR+13;
  const names=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  const angle=i=>(i/12)*2*Math.PI-Math.PI/2;
  const segMid=i=>angle(i)+(angle(1)-angle(0))/2;

  return(
    <svg width={size} height={size} style={{overflow:'visible'}}>
      {[...Array(12)].map((_,i)=>{
        const a1=angle(i),a2=angle(i+1);
        const [ox1,oy1]=[cx+outerR*Math.cos(a1),cy+outerR*Math.sin(a1)];
        const [ox2,oy2]=[cx+outerR*Math.cos(a2),cy+outerR*Math.sin(a2)];
        const [ix1,iy1]=[cx+innerR*Math.cos(a1),cy+innerR*Math.sin(a1)];
        const [ix2,iy2]=[cx+innerR*Math.cos(a2),cy+innerR*Math.sin(a2)];
        const fill=highlighted[i]||'#fff';
        return(
          <path key={i}
            d={`M${ix1},${iy1}L${ox1},${oy1}A${outerR},${outerR} 0 0,1 ${ox2},${oy2}L${ix2},${iy2}A${innerR},${innerR} 0 0,0 ${ix1},${iy1}Z`}
            fill={fill} stroke="#bbb" strokeWidth="0.6"
          />
        );
      })}
      <circle cx={cx} cy={cy} r={innerR} fill="#1a1a1a"/>
      {[...Array(12)].map((_,i)=>{
        const ma=segMid(i);
        const [mx,my]=[cx+midR*Math.cos(ma),cy+midR*Math.sin(ma)];
        return<text key={i} x={mx} y={my+3.5} textAnchor="middle" fontSize={Math.round(size*0.058)} fill="#fff" fontFamily="Raleway,sans-serif">{i}</text>;
      })}
      {[...Array(12)].map((_,i)=>{
        const ma=segMid(i);
        const [lx,ly]=[cx+labelR*Math.cos(ma),cy+labelR*Math.sin(ma)];
        const hi=highlighted[i];
        return<text key={i} x={lx} y={ly+4} textAnchor="middle" fontSize={Math.round(size*0.06)}
          fill={hi&&hi!=='#fff'?hi:C.black} fontFamily="Raleway,sans-serif" fontWeight={hi?'700':'400'}>{names[i]}</text>;
      })}
      {dots.map((d,i)=>{
        const ma=segMid(d.pc);
        const [dx,dy]=[cx+(outerR-6)*Math.cos(ma),cy+(outerR-6)*Math.sin(ma)];
        return<circle key={i} cx={dx} cy={dy} r={size*0.042} fill={d.color} stroke="#fff" strokeWidth="1"/>;
      })}
      {arcs.map((arc,i)=>{
        const ma1=segMid(arc.from),ma2=segMid(arc.to);
        const [ax1,ay1]=[cx+(outerR-4)*Math.cos(ma1),cy+(outerR-4)*Math.sin(ma1)];
        const [ax2,ay2]=[cx+(outerR-4)*Math.cos(ma2),cy+(outerR-4)*Math.sin(ma2)];
        return<path key={i} d={`M${ax1},${ay1}Q${cx},${cy*0.5} ${ax2},${ay2}`}
          fill="none" stroke={arc.color||C.dgreen} strokeWidth="1.8"/>;
      })}
    </svg>
  );
};

// ─── Binary Vector ───────────────────────────────────────────────────────────
const BinaryVector = ({values=[],colors={},separatorAt=null,labels=[],cellW=30}) => {
  const cellH=cellW;
  return(
    <div>
      <div style={{display:'flex'}}>
        {values.map((v,i)=>(
          <div key={i} style={{
            width:cellW,height:cellH,
            border:'1px solid #bbb',
            borderLeft: i===0?'1px solid #bbb': separatorAt===i?'2px solid '+C.red:'none',
            display:'flex',alignItems:'center',justifyContent:'center',
            fontSize:Math.round(cellW*0.42),
            fontWeight:v===1||v==='1'?700:400,
            color:colors[i]||C.black,
            fontFamily:"'JetBrains Mono',monospace",
            background:'#fff',
          }}>{v}</div>
        ))}
      </div>
      {labels.length>0&&(
        <div style={{display:'flex'}}>
          {labels.map((l,i)=>(
            <div key={i} style={{width:cellW,textAlign:'center',fontSize:Math.round(cellW*0.3),
              color:'#888',fontFamily:"'JetBrains Mono',monospace"}}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Roughness Profile (12-bar) ──────────────────────────────────────────────
const RoughnessProfile = ({values=[],barColor=C.blue,width=360,height=120,labels=true}) => {
  const n=values.length||12;
  const maxV=Math.max(...values,0.001);
  const bW=Math.floor(width/n)-2;
  return(
    <svg width={width} height={height+20} style={{overflow:'visible'}}>
      {values.map((v,i)=>{
        const bH=Math.round((v/maxV)*(height-10));
        const x=i*(bW+2)+1;
        const y=height-bH;
        return(
          <g key={i}>
            <rect x={x} y={y} width={bW} height={bH} fill={barColor} opacity="0.85" rx="1"/>
            {labels&&<text x={x+bW/2} y={height+14} textAnchor="middle" fontSize="9"
              fill="#666" fontFamily="Raleway,sans-serif">{i+1}</text>}
          </g>
        );
      })}
      {/* baseline */}
      <line x1="0" y1={height} x2={width} y2={height} stroke="#bbb" strokeWidth="1"/>
    </svg>
  );
};

// ─── Arrow SVG helpers ───────────────────────────────────────────────────────
const ArrowH = ({color=C.black,length=80,label='',labelColor=null,bold=false}) => (
  <svg width={length+10} height={24} style={{overflow:'visible'}}>
    <defs>
      <marker id={`ah-${color.replace('#','')}-${length}`} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
        <path d="M0,0 L0,7 L7,3.5 Z" fill={color}/>
      </marker>
    </defs>
    <line x1="0" y1="12" x2={length} y2="12" stroke={color} strokeWidth={bold?2.5:1.8}
      markerEnd={`url(#ah-${color.replace('#','')}-${length})`}/>
    {label&&<text x={length/2} y={9} textAnchor="middle" fontSize="11" fill={labelColor||color}
      fontWeight="700" fontFamily="Raleway,sans-serif">{label}</text>}
  </svg>
);

// Export everything
Object.assign(window, {
  C, Row, Col, Appear, SoundBtn, SlideChrome,
  PianoKeyboard, ChromaticCircle, BinaryVector, RoughnessProfile, ArrowH,
  IS_BLACK,
});
