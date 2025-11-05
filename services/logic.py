import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from gen.generate import gen_total
from tools.experiment_inversions import run_experiment_with_args


class ExplorationLogic:
    def __init__(self):
        pass

    def generate_chords(self, generation_params):
        """
        Generates chords based on the given parameters.

        :param generation_params: A dictionary with keys like 'pitch_classes',
                                  'o_min', 'o_max', 'N', 'edge_pc0', 'early_filters'.
        :return: A list of generated chords.
        """
        pitch_classes = generation_params.get('pitch_classes', set())
        o_min = generation_params.get('o_min', 4)
        o_max = generation_params.get('o_max', 5)
        N = generation_params.get('N', [3])
        edge_pc0 = generation_params.get('edge_pc0', False)
        early_filters = generation_params.get('early_filters', None)

        generated_chords = list(gen_total(
            pitch_classes=pitch_classes,
            o_min=o_min,
            o_max=o_max,
            N=N,
            edge_pc0=edge_pc0,
            early_filters=early_filters
        ))
        return generated_chords

    def filter_chords(self, chords, filters):
        # Placeholder for filtering logic
        pass

    def run_report(self, report_params: dict, df_override: Optional[pd.DataFrame] = None):
        """
        Runs the experiment and generates the HTML report.

        :param report_params: A dictionary containing the parameters for the report.
        :param df_override: An optional DataFrame to use instead of generating from source.
        """
        args = argparse.Namespace(
            out=Path(report_params.get("output_dir", "outputs/exploration_studio")),
            type=report_params.get("chord_type", "B"),
            query=report_params.get("base_query", None),
            reduction=report_params.get("reduction", "MDS"),
            pops=report_params.get("population_sources", None),
            pops_csv=None,
            pops_file=None,
            model=report_params.get("model", "Sethares"),
            metric=report_params.get("metric", "euclidean"),
            ponderation=report_params.get("ponderation", "ninguna"),
            data_source="generator" if df_override is not None else "database",
            generator_settings=report_params.get("generator_settings")
        )

        return run_experiment_with_args(args, df_override=df_override)
