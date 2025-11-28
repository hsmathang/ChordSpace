import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from tools.proposals_pipeline.metrics import ScenarioEvaluator, ScenarioResult
from tools.proposals_pipeline.population import ChordEntry

class TestScenarioEvaluator(unittest.TestCase):
    def setUp(self):
        # Mock ChordEntry
        # ChordEntry expects specific fields. I'll mock the object to avoid init issues
        # or use a simplified mock class if needed.
        # But `ChordEntry` is a dataclass, so I must match fields.

        # Helper to create mock entries
        def create_mock_entry(n_notes=2):
            m = MagicMock(spec=ChordEntry)
            m.n_notes = n_notes
            return m

        self.entries = [
            create_mock_entry(2),
            create_mock_entry(2)
        ]
        self.preproc_cache = {
            "dummy_preproc": np.array([[0.0, 1.0], [1.0, 0.0]])
        }
        self.dist_simplex_cache = {
            "dummy_preproc": np.array([[0.1, 0.9], [0.8, 0.2]])
        }
        self.distance_cache = {
            ("dummy_preproc", "euclidean"): np.array([0.5]) # Condensed distance for 2 points
        }

        self.context = {
            "entries": self.entries,
            "preproc_cache": self.preproc_cache,
            "dist_simplex_cache": self.dist_simplex_cache,
            "distance_cache": self.distance_cache
        }
        self.evaluator = ScenarioEvaluator(self.context)

    @patch('tools.proposals_pipeline.metrics.compute_embeddings')
    def test_evaluate(self, mock_compute_embeddings):
        # Setup mock return for embedding
        mock_embedding = np.array([[0,0], [1,1]])
        mock_compute_embeddings.return_value = (mock_embedding, {"param": 1})

        task = {
            "scenario": {
                "name": "test_scenario",
                "metric": "euclidean",
                "preproc_id": "dummy_preproc",
                "description": "test"
            },
            "reductions": ["MDS"],
            "seed_list": [42],
            "deterministic": True,
            "jobs": 1,
            "mds_n_init": 1
        }

        result = self.evaluator.evaluate(task)

        self.assertIsInstance(result, ScenarioResult)
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.results[0]["scenario"], "MDS:test_scenario")
        self.assertEqual(len(result.figure_payloads), 1)
        self.assertFalse(result.warnings)

if __name__ == '__main__':
    unittest.main()
