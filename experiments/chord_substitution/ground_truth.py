"""
ground_truth.py — Classical chord substitution pairs from music theory.
Each substitution is defined as (query_type, query_root, sub_type, sub_root, category, explanation).
Roots are pitch classes 0-11 (C=0, Db=1, ..., B=11).
"""

# Substitution categories
TRITONE_SUB = "Tritone Sub"
RELATIVE = "Relative"
PARALLEL = "Parallel"
MEDIANTE = "Chromatic Mediant"
DOMINANT_PREP = "Dominant Prep"
DIMINISHED_SUB = "Diminished Sub"
NEAPOLITAN = "Neapolitan"
COMMON_TONE = "Common Tone"
BORROWED = "Modal Borrowing"

def _generate_all_roots(template):
    """Generate substitution pairs for all 12 roots from a template at root=0."""
    pairs = []
    for root_offset in range(12):
        for qt, qr, st, sr, cat, expl in template:
            pairs.append((
                qt, (qr + root_offset) % 12,
                st, (sr + root_offset) % 12,
                cat, expl
            ))
    return pairs

# Template at root C (0)
_TEMPLATE = [
    # Relative minor/major
    ('maj', 0, 'min', 9, RELATIVE, "Cmaj -> Am: share notes C, E (2/3 common tones)"),
    ('min', 0, 'maj', 3, RELATIVE, "Cm -> Eb: relative major, share Eb, G"),

    # Parallel major/minor
    ('maj', 0, 'min', 0, PARALLEL, "Cmaj -> Cm: modal interchange, share C, G"),

    # Tritone substitution (dominant 7th)
    ('dom7', 0, 'dom7', 6, TRITONE_SUB, "C7 -> Gb7: share tritone E-Bb / E-A#"),

    # ii-V equivalence
    ('min7', 0, 'dom7', 5, DOMINANT_PREP, "Cm7 -> F7: ii-V relationship in Bb"),

    # Chromatic mediants (major thirds apart)
    ('maj', 0, 'maj', 4, MEDIANTE, "Cmaj -> Emaj: upper chromatic mediant"),
    ('maj', 0, 'maj', 8, MEDIANTE, "Cmaj -> Abmaj: lower chromatic mediant"),
    ('maj', 0, 'min', 4, MEDIANTE, "Cmaj -> Em: diatonic mediant"),

    # Diminished substitution for dominant
    ('dom7', 0, 'dim7', 11, DIMINISHED_SUB, "C7 -> Bdim7: share B-D-F tritone"),

    # Neapolitan (bII for V)
    ('dom7', 0, 'maj', 1, NEAPOLITAN, "C7 -> Dbmaj: Neapolitan sub (bII of C)"),

    # Common-tone diminished
    ('maj', 0, 'dim7', 1, COMMON_TONE, "Cmaj -> C#dim7: common-tone C, approach from above"),

    # Modal borrowing
    ('maj', 0, 'maj', 8, BORROWED, "Cmaj -> Abmaj: bVI borrowed from minor"),
    ('maj', 0, 'min', 5, BORROWED, "Cmaj -> Fm: iv borrowed from minor mode"),
]

SUBSTITUTION_PAIRS = _generate_all_roots(_TEMPLATE)

def get_substitution_pairs():
    """Return all substitution pairs as list of dicts."""
    return [
        {
            'query_type': qt, 'query_root': qr,
            'sub_type': st, 'sub_root': sr,
            'category': cat, 'explanation': expl,
        }
        for qt, qr, st, sr, cat, expl in SUBSTITUTION_PAIRS
    ]

def build_ground_truth_set(corpus):
    """Build a set of (query_name, sub_name) pairs for fast lookup.
    corpus is a list of dicts with 'name', 'type', 'root' keys."""
    from common import NOTE_NAMES
    lookup = {}
    for c in corpus:
        key = (c['type'], c['root'])
        lookup[key] = c['name']

    gt_set = set()
    for p in get_substitution_pairs():
        qkey = (p['query_type'], p['query_root'])
        skey = (p['sub_type'], p['sub_root'])
        if qkey in lookup and skey in lookup:
            gt_set.add((lookup[qkey], lookup[skey]))
            gt_set.add((lookup[skey], lookup[qkey]))  # symmetric
    return gt_set

def get_substitution_categories():
    """Return list of all category names."""
    return [TRITONE_SUB, RELATIVE, PARALLEL, MEDIANTE,
            DOMINANT_PREP, DIMINISHED_SUB, NEAPOLITAN, COMMON_TONE, BORROWED]
