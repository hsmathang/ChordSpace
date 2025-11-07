"""
Benchmark different chord representation strategies for combinatorial generation.

The script compares three approaches:
1. Dual record (normalized mask + real-world metadata/frequencies).
2. Normalized mask + root offset only.
3. Parametric (root MIDI + interval vector).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synth_tools import _build_record_from_notes

A4_HZ = 440.0


def midi_to_freq(midi_note: int) -> float:
    """Return frequency in Hz for a MIDI note (12-TET)."""
    return A4_HZ * (2.0 ** ((midi_note - 69) / 12.0))


def build_midi_universe(alphabet: Sequence[int], octave_min: int, octave_max: int) -> List[int]:
    universe: List[int] = []
    for octave in range(octave_min, octave_max + 1):
        for pc in alphabet:
            midi_note = 12 * (octave + 1) + pc
            if 0 <= midi_note <= 127:
                universe.append(midi_note)
    return sorted(set(universe))


def generate_chords(
    alphabet: Sequence[int],
    octave_min: int,
    octave_max: int,
    cardinalities: Sequence[int],
) -> List[Tuple[int, ...]]:
    universe = build_midi_universe(alphabet, octave_min, octave_max)
    chords: List[Tuple[int, ...]] = []
    for k in cardinalities:
        if k <= 0 or k > len(universe):
            continue
        for combo in itertools.combinations(universe, k):
            chords.append(tuple(combo))
    return chords


def dual_record_repr(chord: Sequence[int]) -> Dict[str, Any]:
    root = chord[0]
    normalized = [note - root for note in chord]
    record = _build_record_from_notes(normalized)
    mask_norm = int(record["abs_mask_int"])
    intervals = tuple(int(x) for x in record["interval"])
    pc_signature = tuple(int(x) for x in record["notes"])
    freqs = [midi_to_freq(n) for n in chord]
    return {
        "root_midi": root,
        "mask_normalized": mask_norm,
        "interval_signature": intervals,
        "pc_signature": pc_signature,
        "frequencies_hz": freqs,
    }


def normalized_mask_repr(chord: Sequence[int]) -> Dict[str, Any]:
    root = chord[0]
    mask = 0
    for note in chord:
        mask |= 1 << (note - root)
    return {"root_midi": root, "mask_normalized": mask, "n": len(chord)}


def parametric_repr(chord: Sequence[int]) -> Dict[str, Any]:
    root = chord[0]
    intervals = tuple(chord[i + 1] - chord[i] for i in range(len(chord) - 1))
    return {"root_midi": root, "intervals": intervals, "n": len(chord)}


def default_key(data: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(sorted(data.items()))


def dual_key(data: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        data["root_midi"],
        data["mask_normalized"],
        data["interval_signature"],
        data["pc_signature"],
    )


def normalized_key(data: Dict[str, Any]) -> Tuple[Any, ...]:
    return (data["root_midi"], data["mask_normalized"], data["n"])


def parametric_key(data: Dict[str, Any]) -> Tuple[Any, ...]:
    return (data["root_midi"], data["intervals"])


StrategyFunc = Callable[[Sequence[int]], Dict[str, Any]]
KeyFunc = Callable[[Dict[str, Any]], Tuple[Any, ...]]


@dataclass(frozen=True)
class Strategy:
    name: str
    builder: StrategyFunc
    key_func: KeyFunc


STRATEGIES: Tuple[Strategy, ...] = (
    Strategy("dual_record", dual_record_repr, dual_key),
    Strategy("normalized_mask", normalized_mask_repr, normalized_key),
    Strategy("parametric", parametric_repr, parametric_key),
)


@dataclass(frozen=True)
class Scenario:
    name: str
    alphabet: Tuple[int, ...]
    octave_min: int
    octave_max: int
    cardinalities: Tuple[int, ...]


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("Diatonic 2 oct (triads)", (0, 2, 4, 5, 7, 9, 11), 3, 4, (3,)),
    Scenario("Diatonic 3 oct (triads+tetra)", (0, 2, 4, 5, 7, 9, 11), 3, 5, (3, 4)),
    Scenario("Modal 3 oct (8 pcs)", (0, 1, 3, 5, 6, 8, 10, 11), 2, 4, (3, 4)),
)


def evaluate_strategy(
    strategy: Strategy,
    chords: Sequence[Tuple[int, ...]],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    serialized_lengths: List[int] = []
    key_set: set[Tuple[Any, ...]] = set()
    for chord in chords:
        data = strategy.builder(chord)
        serialized_lengths.append(len(json.dumps(data, default=float)))
        key_set.add(strategy.key_func(data))
    elapsed = time.perf_counter() - t0
    avg_bytes = statistics.mean(serialized_lengths) if serialized_lengths else 0.0
    return {
        "strategy": strategy.name,
        "elapsed_sec": elapsed,
        "per_chord_ms": (elapsed / len(chords) * 1000.0) if chords else 0.0,
        "approx_bytes_per_chord": avg_bytes,
        "unique_keys": len(key_set),
        "collisions": len(chords) - len(key_set),
    }


def run_scenario(scenario: Scenario) -> Dict[str, Any]:
    chords = generate_chords(
        scenario.alphabet,
        scenario.octave_min,
        scenario.octave_max,
        scenario.cardinalities,
    )
    stats = [evaluate_strategy(strategy, chords) for strategy in STRATEGIES]
    return {
        "scenario": scenario.name,
        "chord_count": len(chords),
        "alphabet": scenario.alphabet,
        "octaves": (scenario.octave_min, scenario.octave_max),
        "cardinalities": scenario.cardinalities,
        "results": stats,
    }


def print_report(data: Dict[str, Any]) -> None:
    print(f"\nScenario: {data['scenario']} | chords={data['chord_count']}")
    print(f"  alphabet={data['alphabet']} octaves={data['octaves']} cardinalities={data['cardinalities']}")
    for item in data["results"]:
        print(
            f"    - {item['strategy']:<15} time={item['elapsed_sec']:.4f}s "
            f"({item['per_chord_ms']:.4f} ms/chord) "
            f"bytes~{item['approx_bytes_per_chord']:.1f} "
            f"unique={item['unique_keys']} collisions={item['collisions']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark chord representation strategies.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of human-readable summary.",
    )
    args = parser.parse_args()

    report_data = [run_scenario(scenario) for scenario in SCENARIOS]
    if args.json:
        print(json.dumps(report_data, indent=2))
    else:
        for item in report_data:
            print_report(item)


if __name__ == "__main__":
    main()
