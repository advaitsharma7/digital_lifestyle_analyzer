from __future__ import annotations

import unittest

from src.training import ensure_artifacts


class ArtifactsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.metadata = ensure_artifacts()

    def test_metadata_contains_expected_sections(self) -> None:
        for key in [
            "defaults",
            "options",
            "metrics",
            "feature_importance",
            "app_usage_estimator",
            "cluster_descriptions",
        ]:
            self.assertIn(key, self.metadata)

    def test_feature_importances_are_normalized(self) -> None:
        for metric in ["stress", "productivity"]:
            total = sum(self.metadata["feature_importance"][metric].values())
            self.assertAlmostEqual(total, 1.0, places=3)

    def test_metrics_are_in_expected_ranges(self) -> None:
        self.assertGreaterEqual(self.metadata["metrics"]["stress_accuracy"], 0.0)
        self.assertLessEqual(self.metadata["metrics"]["stress_accuracy"], 1.0)
        self.assertGreaterEqual(self.metadata["metrics"]["productivity_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
