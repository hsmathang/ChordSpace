import pytest
from unittest.mock import patch
import pandas as pd
from services.logic import ExplorationLogic

def test_generate_chords_diatonic_triads():
    logic = ExplorationLogic()
    generation_params = {
        'pitch_classes': {0, 2, 4, 5, 7, 9, 11},
        'o_min': 4,
        'o_max': 5,  # Corrected to include the necessary octave for G major
        'N': [3]
    }

    chords = logic.generate_chords(generation_params)

    assert isinstance(chords, list)
    assert len(chords) > 0

    # C major triad
    c_major = (60, 64, 67)
    assert any(c.midi == c_major for c in chords), "C major triad not found"

    # G major triad
    g_major = (67, 71, 74)
    assert any(c.midi == g_major for c in chords), "G major triad not found"

@patch('services.logic.run_experiment_with_args')
def test_run_report(mock_run_experiment):
    logic = ExplorationLogic()

    # Create a dummy DataFrame of chords
    dummy_chords = [
        {'midi': (60, 64, 67), 'n': 3, 'span': 7, 'pc_mask': 145},
        {'midi': (62, 65, 69), 'n': 3, 'span': 7, 'pc_mask': 548}
    ]
    df_override = pd.DataFrame(dummy_chords)

    report_params = {
        "output_dir": "test_output",
        "reduction": "UMAP",
        "model": "Euler",
        "metric": "cosine",
        "ponderation": "consonancia",
        "comparison_seeds": "123"
    }

    logic.run_report(report_params, df_override=df_override)

    mock_run_experiment.assert_called_once()
    args, kwargs = mock_run_experiment.call_args

    # Check that the argparse.Namespace object has the correct attributes
    namespace = args[0]
    assert namespace.out.name == "test_output"
    assert namespace.reduction == "UMAP"
    assert namespace.model == "Euler"
    assert namespace.metric == "cosine"
    assert namespace.ponderation == "consonancia"

    # Check that the df_override is passed correctly
    pd.testing.assert_frame_equal(kwargs['df_override'], df_override)
