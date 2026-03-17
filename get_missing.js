const fs = require('fs');
const texFile = 'docs/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho/01Seccion01.tex';
const bibFile = 'docs/Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho/ReferenciasRugosas.bib';

try {
    const tex = fs.readFileSync(texFile, 'utf8');
    const bib = fs.readFileSync(bibFile, 'utf8');

    const cites = new Set();
    const regexTex = /\\cite\{([^}]+)\}/g;
    let match;
    while ((match = regexTex.exec(tex)) !== null) {
        const keys = match[1].split(',').map(k => k.trim());
        keys.forEach(k => cites.add(k));
    }

    const bibKeys = new Set();
    const regexBib = /@\w+\s*\{([^,]+)/g;
    while ((match = regexBib.exec(bib)) !== null) {
        bibKeys.add(match[1].trim());
    }

    const missing = [...cites].filter(x => !bibKeys.has(x));

    fs.writeFileSync('out.txt', `Missing: ${missing.join(', ')}\nTotal found in tex: ${cites.size}\n`);
    console.log("Success");
} catch (e) {
    fs.writeFileSync('out.txt', e.toString());
    console.log("Error", e);
}
