$keys = "piston1962harmony","chew2014mathematical","bigo2014self","abraham1987historia","conservatorioSalamanca_undated","cook2009harmony","cheng2008automatic","forte1973structure","krumhansl1990cognitive","sethares2005tuning","parncutt1989harmony","helmholtz1875sensations","vassilakis2001perceptual","milne2023universal","masina2022dyads","masina2023triads","Bowling2018VocalSimilarity","cuthbert2010music21","burgoyne2011expert","jimenez2020common","barthet2011music","bernardes2016multi","cambouropoulos2016harmonic","madjiheurem2016chord2vec","lerdahl1988tonal","tymoczko2006geometry","tymoczko2012generalized","Nardelli","callenderGeneralizedVoiceLeadingSpaces2008","himpelGeometryMusicPerception2022","euler1739tentamen","krumhansl1990","Tymoczko2016GeometryOM","cohn1997transformational","lozano2009generation","huang2016chordripple","xambo2018jam","navarro2020assistive","plomp1965tonal","sethares1993local","hutchinson1978acoustical","bowling2018vocal","wilk2019automatic","borg2005modern","mcinnes2018umap","coenen2019understanding"

$bibContent = Get-Content "d:\Documents\GitHub\ChordSpace\docs\Tesis_Maestría_Matemáticas_Aplicadas_UNAL_2024 pacho\ReferenciasRugosas.bib" -Raw

$missing = @()
foreach ($key in $keys) {
    if ($bibContent -notmatch "@[a-zA-Z]+\{$key,") {
        $missing += $key
    }
}

$missing | Out-File "d:\Documents\GitHub\ChordSpace\missing_cites.txt" -Encoding utf8
