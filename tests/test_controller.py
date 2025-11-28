import unittest
from unittest.mock import MagicMock, patch
import queue
import threading
import pandas as pd
from ui.launcher.controller import ExperimentController
from services.domain import ExperimentConfig, VisualizationConfig, ExperimentResult, RoughnessConfig, ReductionConfig, ExecutionConfig

class TestExperimentController(unittest.TestCase):
    def setUp(self):
        self.log_queue = queue.Queue()
        self.controller = ExperimentController(self.log_queue)

    def test_generate_combinatorial_population(self):
        # Mocking the generator function to avoid actual computation
        with patch('ui.launcher.controller.generate_combinatorial_chords') as mock_gen:
            mock_df = pd.DataFrame({'id': [1], 'notes': [[60]]})
            mock_gen.return_value = mock_df

            df = self.controller.generate_combinatorial_population(
                alphabet=[0, 4, 7], octave_min=3, octave_max=4,
                cardinalities=[3], structural_mode=False
            )

            self.assertEqual(len(df), 1)
            self.assertTrue(self.controller.temporal_population_df is not None)
            mock_gen.assert_called_once()

    def test_add_temporal_to_final(self):
        self.controller.temporal_population_df = pd.DataFrame({'id': [1], 'n': [3]})
        self.controller.add_temporal_to_final()

        self.assertEqual(len(self.controller.final_population_df), 1)

        # Add again
        self.controller.temporal_population_df = pd.DataFrame({'id': [2], 'n': [3]})
        self.controller.add_temporal_to_final()
        self.assertEqual(len(self.controller.final_population_df), 2)

    def test_clear_final_population(self):
        self.controller.final_population_df = pd.DataFrame({'id': [1]})
        self.controller.clear_final_population()
        self.assertIsNone(self.controller.final_population_df)

    @patch('ui.launcher.controller.ExperimentService')
    @patch('ui.launcher.controller.VisualizationService')
    def test_run_experiment_async(self, MockVisService, MockExpService):
        # Setup mocks
        mock_exp_service_instance = MockExpService.return_value
        mock_vis_service_instance = MockVisService.return_value

        # Re-init controller to inject mocks
        self.controller.experiment_service = mock_exp_service_instance
        self.controller.visualization_service = mock_vis_service_instance

        mock_result = ExperimentResult(
            config=MagicMock(), population_df=pd.DataFrame(), metrics_df=pd.DataFrame(),
            embeddings={}, timing=[], output_path=None
        )
        mock_exp_service_instance.run_experiment.return_value = mock_result
        mock_vis_service_instance.generate_report_full.return_value = "report.html"

        # Construct a real or fully mocked config
        config = ExperimentConfig(
            population=MagicMock(),
            roughness=RoughnessConfig(proposals=[], metrics=[]),
            reduction=ReductionConfig(methods=[]),
            execution=ExecutionConfig(seeds=[]),
            name="test_experiment"
        )

        vis_config = MagicMock(spec=VisualizationConfig)

        # Run
        thread = self.controller.run_experiment_async(config, vis_config)
        thread.join()

        # Verify
        mock_exp_service_instance.run_experiment.assert_called_once()
        mock_vis_service_instance.generate_report_full.assert_called_once()

        # Check logs
        msgs = []
        while not self.log_queue.empty():
            msgs.append(self.log_queue.get())

        self.assertTrue(any(k == 'done' for k, v in msgs))
        self.assertTrue(any(k == 'compare_status' and v == 'Completado' for k, v in msgs))

if __name__ == '__main__':
    unittest.main()
