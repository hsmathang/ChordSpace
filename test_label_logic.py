from tools.proposals_pipeline.figures import _generate_compact_labels
from tools.proposals_pipeline.population import ChordEntry
from dataclasses import dataclass
from typing import List

@dataclass
class MockAcorde:
    notes_abs: List[int]
    intervals: List[int]

@dataclass
class MockEntry:
    acorde: MockAcorde

# Test Case 1: C Major (0, 4, 7)
c_maj = MockEntry(MockAcorde([60, 64, 67], [4, 3]))
# Normalized: 0, 4, 7 -> "047". Intervals: [4, 3]

# Test Case 2: C Major Inversion (G, C, E) -> 67, 72, 76.
# PCs: 7, 0, 4. Sorted: 0, 4, 7. Normalized: 0, 4, 7 -> "047".
# Intervals: [5, 4]
c_inv = MockEntry(MockAcorde([67, 72, 76], [5, 4]))

# Test Case 3: 0257 example from user.
# Try to match user example: 0257 [2,5,2].
# Intervals 2, 5, 2 -> 0, 2, 7, 9.
# Notes: 60, 62, 67, 69.
# PCs: 0, 2, 7, 9. Norm: 0279.
# Note: User said "0257" but math says "0279". I will output what the logic produces.
test_0257 = MockEntry(MockAcorde([60, 62, 67, 69], [2, 5, 2]))

# Test Case 4: A and B (10, 11).
# C7 dominant (C, E, G, Bb) -> 0, 4, 7, 10.
# Norm: 047A.
c7 = MockEntry(MockAcorde([60, 64, 67, 70], [4, 3, 3]))

entries = [c_maj, c_inv, test_0257, c7]

try:
    labels = _generate_compact_labels(entries)
    for l in labels:
        print(l)
except ImportError:
    print("Function not found yet")
except Exception as e:
    print(e)
