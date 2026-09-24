import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from churnguard.evaluation import ranking_metrics


class RankingMetricTests(unittest.TestCase):
    def test_constant_score_baseline_has_neutral_lift(self):
        result = ranking_metrics([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(result["lift_at_decile_1"], 1.0)

    def test_lift_handles_ties_at_decile_boundary_without_row_order_bias(self):
        result = ranking_metrics([1, 0, 1, 0, 1, 0, 0, 1, 0, 0], [0.9] * 4 + [0.1] * 6)
        self.assertAlmostEqual(result["lift_at_decile_1"], 1.25)


if __name__ == "__main__":
    unittest.main()
