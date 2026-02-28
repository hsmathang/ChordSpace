#!/usr/bin/env python3
"""
consolidar_bib.py  v2  (usa bibtexparser — mucho más rápido)
=============================================================
Consolida múltiples .bib en ReferenciasRugosas.bib:
  1. Cargar por orden de confianza
  2. Filtro de calidad (rechaza TBD / sin año / sin datos mínimos)
  3. Deduplicar: DOI → clave → título (índice invertido de tokens)
  4. Limpiar: quitar abstract/file/langid, normalizar date→year,
              journaltitle→journal, rutas locales, title {{}} → {}
  5. Renombrar claves: apellido+año+keynword  (ej. vassilakis2001perceptual)
  6. Escribir ReferenciasRugosas.bib + consolidacion_report.md
"""

import os
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict

try:
    import bibtexparser
    from bibtexparser.bparser import BibTexParser
    from bibtexparser.customization import convert_to_unicode
except ImportError:
    sys.exit("ERROR: bibtexparser no instalado. Ejecuta:  pip install bibtexparser")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = r"d:\Documents\GitHub\ChordSpace"
TESIS_DIR = os.path.join(REPO_ROOT,
    r"docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho")

# Archivos en orden DECRECIENTE de confianza (gana el de menor índice)
SOURCE_FILES = [
    os.path.join(REPO_ROOT, r"docs\ZOTERO\Tesis MSc UNAL.bib"),
    os.path.join(REPO_ROOT, r"docs\thesis\referencias_metodologia_vf.bib"),
    os.path.join(REPO_ROOT, r"metodologia_temporal\Tesis MSc UNALvf\Tesis MSc UNALvf.bib"),
    os.path.join(REPO_ROOT, r"docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\Referencias.bib"),
    os.path.join(REPO_ROOT, r"docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\unified_references.bib"),
    os.path.join(REPO_ROOT, r"docs\thesis\referencias_metodologia.bib"),
]

OUTPUT_FILE  = os.path.join(TESIS_DIR, "ReferenciasRugosas.bib")
REPORT_FILE  = os.path.join(TESIS_DIR, "consolidacion_report.md")

# Campos a ELIMINAR del output (innecesarios para LaTeX /BibTeX clásico)
FIELDS_REMOVE = {
    "abstract", "file", "langid", "copyright", "keywords",
    "pubstate", "archiveprefix", "primaryclass", "eprinttype",
    "eprintclass", "date-added", "date-modified", "urldate",
    "journaltitle", "shortjournal", "eventtitle", "location",
    "namea", "nameatype", "pagetotal", "annotation",
}

# Campos permitidos en el output final
FIELDS_KEEP = {
    "author", "title", "year", "journal", "booktitle", "volume",
    "number", "pages", "publisher", "address", "doi", "url",
    "isbn", "issn", "series", "edition", "editor", "school",
    "institution", "chapter", "organization", "howpublished",
    "type", "eprint", "month",
}

STOPWORDS = {
    "a","an","the","of","in","on","for","and","to","with","is","are",
    "was","were","by","from","as","at","be","its","this","that","or",
    "not","de","la","el","en","del","y","por","con","una","para",
}

# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def normalize_text(text):
    text = strip_accents(text.lower().strip())
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text

def title_sim(t1, t2):
    return SequenceMatcher(None, normalize_text(t1), normalize_text(t2)).ratio()

def extract_last_name(author):
    if not author:
        return "unknown"
    first = re.split(r"\s+and\s+", author, flags=re.IGNORECASE)[0].strip()
    first = re.sub(r"[{}\\]", "", first).strip()
    if "," in first:
        last = first.split(",")[0].strip()
    else:
        parts = first.split()
        last = parts[-1] if parts else "unknown"
    last = re.sub(r"[^a-z]", "", normalize_text(last))
    return last or "unknown"

def extract_year(entry):
    for field in ("year", "date"):
        val = entry.get(field, "")
        m = re.search(r"\d{4}", str(val))
        if m:
            return m.group(0)
    return ""

def title_keyword(title):
    clean = re.sub(r"[{}\\]", "", title or "")
    words = normalize_text(clean).split()
    for w in words:
        if w not in STOPWORDS and len(w) > 2 and w.isalpha():
            return w
    return words[0] if words else "nokey"

def make_key(entry):
    """Genera clave apellido+año+keyword, ej. vassilakis2001perceptual."""
    last = extract_last_name(entry.get("author", ""))
    year = extract_year(entry)
    kw   = title_keyword(entry.get("title", ""))
    return f"{last}{year}{kw}"

def extract_doi(entry):
    doi = entry.get("doi", "").strip().lower()
    doi = re.sub(r"^https?://doi\.org/", "", doi)
    doi = re.sub(r"[{}]", "", doi)
    return doi

def completeness(entry):
    score = 0
    if entry.get("author"):  score += 2
    if entry.get("title"):   score += 2
    if extract_year(entry):  score += 1
    if entry.get("doi"):     score += 3
    if entry.get("journal") or entry.get("booktitle"): score += 1
    if entry.get("pages"):   score += 1
    return score

def get_tokens(title, min_len=4):
    words = normalize_text(re.sub(r"[{}\\]", "", title)).split()
    return set(w for w in words if len(w) >= min_len and w not in STOPWORDS)

# ─────────────────────────────────────────────────────────────────────────────
# CARGAR ARCHIVOS
# ─────────────────────────────────────────────────────────────────────────────

def load_bib(filepath):
    try:
        parser = BibTexParser(common_strings=True)
        parser.customization = convert_to_unicode
        parser.ignore_nonstandard_types = False
        with open(filepath, encoding="utf-8", errors="replace") as f:
            db = bibtexparser.load(f, parser=parser)
        return db.entries
    except Exception as e:
        print(f"  ⚠️  Error al cargar {filepath}: {e}")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# FILTRO DE CALIDAD
# ─────────────────────────────────────────────────────────────────────────────

def quality_ok(entry):
    """Retorna (ok, reason)."""
    title  = entry.get("title", "").strip()
    author = entry.get("author", "").strip()
    doi    = entry.get("doi", "").strip()
    url    = entry.get("url", "").strip()
    year   = extract_year(entry)

    if re.search(r"\bTBD\b|TODO|PLACEHOLDER|TBD:", title, re.I):
        return False, "TBD en título"
    if not title:
        return False, "Sin título"
    if not year:
        return False, "Sin año"
    if not author and not doi and not url:
        return False, "Sin autor+doi+url"
    return True, "OK"

# ─────────────────────────────────────────────────────────────────────────────
# LIMPIEZA DE ENTRADAS
# ─────────────────────────────────────────────────────────────────────────────

def clean(entry):
    e = dict(entry)

    # date → year (BibLaTeX → BibTeX)
    if "date" in e and "year" not in e:
        m = re.search(r"\d{4}", e["date"])
        if m:
            e["year"] = m.group(0)

    # journaltitle → journal
    if "journaltitle" in e and "journal" not in e:
        e["journal"] = e["journaltitle"]

    # location → address
    if "location" in e and "address" not in e:
        e["address"] = e["location"]

    # eventtitle → booktitle
    if "eventtitle" in e and "booktitle" not in e:
        e["booktitle"] = e["eventtitle"]

    # Eliminar URL si es ruta local
    if "url" in e and re.match(r"[A-Za-z]:\\|/home/|/Users/", e.get("url", "")):
        del e["url"]

    # Limpiar {{Title}} → {Title} en campo title
    if "title" in e:
        e["title"] = re.sub(r"\{\{([^}]+)\}\}", r"{\1}", e["title"])

    # Quitar campos innecesarios
    for f in list(e.keys()):
        if f in FIELDS_REMOVE or (f not in FIELDS_KEEP and
                                   f not in ("ENTRYTYPE", "ID")):
            e.pop(f, None)

    return e

# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(ordered):
    """
    ordered: list de (entry_dict, rank)  (rank 0 = más confiable)
    Retorna (accepted_list, dup_log)
    """
    accepted   = []
    dup_log    = []
    doi_idx    = {}          # doi  → índice en accepted
    key_idx    = {}          # key  → índice en accepted
    tok_idx    = defaultdict(set)  # token → set de índices

    for entry, rank in ordered:
        doi   = extract_doi(entry)
        key   = entry.get("ID", "unknown")
        title = entry.get("title", "")

        # 1) Por DOI
        if doi:
            if doi in doi_idx:
                i = doi_idx[doi]
                if completeness(entry) > completeness(accepted[i]):
                    accepted[i] = entry
                dup_log.append(f"DUP_DOI  | `{key}` ≡ `{accepted[i].get('ID','')}` (DOI: {doi})")
                continue
            doi_idx[doi] = len(accepted)

        # 2) Por clave exacta
        if key in key_idx:
            i = key_idx[key]
            if completeness(entry) > completeness(accepted[i]):
                accepted[i] = entry
            dup_log.append(f"DUP_KEY  | `{key}` (clave duplicada)")
            continue

        # 3) Por similitud de título con índice de tokens
        found = False
        if title:
            tokens = get_tokens(title)
            votes  = defaultdict(int)
            for t in tokens:
                for i in tok_idx[t]:
                    votes[i] += 1
            for i, v in votes.items():
                if v < 2:
                    continue
                sim = title_sim(title, accepted[i].get("title", ""))
                if sim >= 0.92:
                    if completeness(entry) > completeness(accepted[i]):
                        old_key = accepted[i].get("ID", "")
                        accepted[i] = entry
                        key_idx.pop(old_key, None)
                        key_idx[key] = i
                        old_tok = get_tokens(accepted[i].get("title", ""))
                        for t in old_tok: tok_idx[t].discard(i)
                        for t in tokens:  tok_idx[t].add(i)
                    dup_log.append(f"DUP_TITLE| `{key}` ~ `{accepted[i].get('ID','')}` (sim={sim:.2f})")
                    found = True
                    break

        if found:
            continue

        # Nueva entrada única
        ni = len(accepted)
        key_idx[key] = ni
        if doi:
            doi_idx[doi] = ni
        if title:
            for t in get_tokens(title):
                tok_idx[t].add(ni)
        accepted.append(entry)

    return accepted, dup_log

# ─────────────────────────────────────────────────────────────────────────────
# RENOMBRAR CLAVES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_keys(entries):
    changes = []
    used    = {}
    for e in entries:
        old = e.get("ID", "unknown")
        new = make_key(e)
        if new in used:
            for suf in "bcdefghijklmnopqrstuvwxyz":
                candidate = new + suf
                if candidate not in used:
                    new = candidate
                    break
        used[new] = old
        if old != new:
            changes.append(f"  {old}  →  {new}")
        e["ID"] = new
    return entries, changes

# ─────────────────────────────────────────────────────────────────────────────
# SERIALIZAR BIB
# ─────────────────────────────────────────────────────────────────────────────

FIELD_ORDER = [
    "author","title","journal","booktitle","year","volume","number",
    "pages","publisher","address","doi","url","isbn","issn","series",
    "edition","editor","school","institution","eprint","month","type",
    "organization","howpublished","chapter",
]

def entry_to_str(e):
    etype = e.get("ENTRYTYPE", "misc")
    key   = e.get("ID", "unknown")
    lines = [f"@{etype}{{{key},"]
    done  = {"ENTRYTYPE","ID"}
    for f in FIELD_ORDER:
        if f in e and f not in done:
            lines.append(f"  {f} = {{{e[f]}}},")
            done.add(f)
    for f in sorted(e.keys()):
        if f not in done:
            lines.append(f"  {f} = {{{e[f]}}},")
    lines.append("}")
    return "\n".join(lines)

def write_bib(entries, path):
    header = (
        "% ReferenciasRugosas.bib\n"
        "% Generado automáticamente — NO editar manualmente\n"
        "% Claves: apellido+año+palabra  (ej. vassilakis2001perceptual)\n"
        f"% Total: {len(entries)} entradas únicas\n\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for e in sorted(entries, key=lambda x: x.get("ID","")):
            f.write(entry_to_str(e))
            f.write("\n\n")
    print(f"  ✅  {path}")
    print(f"      {len(entries)} entradas.")

# ─────────────────────────────────────────────────────────────────────────────
# REPORTE
# ─────────────────────────────────────────────────────────────────────────────

def write_report(path, accepted, rejected, dups, changes, no_doi):
    rows = "\n".join(
        f"| `{e.get('ID','')}` | {e.get('ENTRYTYPE','')} | "
        f"{extract_year(e)} | {e.get('author','—')[:40]} |"
        for e in sorted(accepted, key=lambda x: x.get("ID",""))
    )
    nodoi_list = "\n".join(
        f"- `{e.get('ID','')}`: *{e.get('title','')[:80]}*"
        for e in no_doi
    )
    rej_list   = "\n".join(f"- `{r['key']}` → {r['reason']}: *{r['title'][:70]}*"
                           for r in rejected)
    dup_list   = "\n".join(f"- {d}" for d in dups)
    chg_list   = "\n".join(changes) if changes else "  (sin cambios)"

    content = f"""# Reporte — ReferenciasRugosas.bib

**Aceptadas:** {len(accepted)}  **Duplicados eliminados:** {len(dups)}  **Rechazadas:** {len(rejected)}  **Sin DOI:** {len(no_doi)}

---

## ✅ Entradas Aceptadas

| Clave | Tipo | Año | Autor |
|-------|------|-----|-------|
{rows}

---

## ⚠️ Sin DOI (verificación recomendada)

{nodoi_list or "— Ninguna —"}

---

## 🔄 Claves Renombradas

```
{chg_list}
```

---

## ❌ Rechazadas por Calidad

{rej_list or "— Ninguna —"}

---

## 🔁 Duplicados Eliminados

{dup_list or "— Ninguno —"}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  📊  {path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  CONSOLIDADOR BibTeX  →  ReferenciasRugosas.bib")
    print("=" * 60)

    # 1. Cargar
    print("\n[1/5] Cargando archivos fuente...")
    all_entries = []
    for rank, fp in enumerate(SOURCE_FILES):
        if not os.path.exists(fp):
            print(f"  [{rank}] ⚠️  No encontrado: {os.path.basename(fp)}")
            continue
        entries = load_bib(fp)
        print(f"  [{rank}] {os.path.basename(fp):50s} {len(entries):4d} entradas")
        for e in entries:
            all_entries.append((e, rank))
    print(f"\n  Total bruto: {len(all_entries)}")

    # 2. Filtrar calidad
    print("\n[2/5] Filtro de calidad...")
    rejected   = []
    good       = []
    for e, rank in all_entries:
        ok, reason = quality_ok(e)
        if ok:
            good.append((e, rank))
        else:
            rejected.append({
                "key"   : e.get("ID","?"),
                "reason": reason,
                "title" : e.get("title","—"),
            })
    print(f"  Aceptadas: {len(good)}  Rechazadas: {len(rejected)}")

    # 3. Deduplicar
    print("\n[3/5] Deduplicando...")
    accepted, dup_log = deduplicate(good)
    print(f"  Únicas: {len(accepted)}  Duplicados: {len(dup_log)}")

    # 4. Limpiar campos
    print("\n[4/5] Limpiando campos...")
    accepted = [clean(e) for e in accepted]

    # 5. Renombrar claves
    print("\n[5/5] Normalizando claves (apellido+año+palabra)...")
    accepted, key_changes = normalize_keys(accepted)
    print(f"  Claves renombradas: {len(key_changes)}")

    no_doi = [e for e in accepted if not e.get("doi")]
    print(f"  Sin DOI (avisar):   {len(no_doi)}")

    # Escribir
    print("\n       Escribiendo archivos...")
    write_bib(accepted, OUTPUT_FILE)
    write_report(REPORT_FILE, accepted, rejected, dup_log, key_changes, no_doi)

    print("\n" + "=" * 60)
    print(f"  ✅ LISTO")
    print(f"  Entradas finales:  {len(accepted)}")
    print(f"  Duplicados:        {len(dup_log)}")
    print(f"  Rechazadas:        {len(rejected)}")
    print(f"  Sin DOI:           {len(no_doi)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
