from __future__ import annotations

import unittest

from src.data import load_processed_dataset
from src.inference import analyze_profile, load_artifacts
from src.training import ensure_artifacts


class InferenceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_artifacts()
        cls.reference_df = load_processed_dataset()
        cls.metadata = load_artifacts()["metadata"]
        cls.sample_inputs = {
            "Daily_Phone_Hours": 6.4,
            "Social_Media_Hours": 3.3,
            "Sleep_Hours": 6.6,
            "Caffeine_Intake_Cups": 2.0,
            "Weekend_Screen_Time_Hours": 8.1,
            "Age": 29,
            "Gender": "Female",
            "Device_Type": "iOS",
        }

    def test_analysis_output_has_expected_shape(self) -> None:
        result = analyze_profile(self.sample_inputs, self.reference_df)
        self.assertIn("global_feature_importance", result)
        self.assertIn("local_impacts", result)
        self.assertIn("insights", result)
        self.assertIn("cluster", result)
        self.assertIn("percentiles", result)
        self.assertGreaterEqual(result["stress_level"], 1)
        self.assertLessEqual(result["stress_level"], 5)
        self.assertGreaterEqual(result["productivity_score"], 1.0)
        self.assertLessEqual(result["productivity_score"], 10.0)

    def test_insights_count_matches_spec(self) -> None:
        result = analyze_profile(self.sample_inputs, self.reference_df)
        self.assertGreaterEqual(len(result["insights"]), 3)
        self.assertLessEqual(len(result["insights"]), 5)

    def test_app_usage_count_stays_within_training_bounds(self) -> None:
        result = analyze_profile(self.sample_inputs, self.reference_df)
        bounds = self.metadata["app_usage_bounds"]
        self.assertGreaterEqual(result["profile"]["App_Usage_Count"], bounds["min"])
        self.assertLessEqual(result["profile"]["App_Usage_Count"], bounds["max"])

    def test_optional_inputs_can_be_omitted(self) -> None:
        result = analyze_profile(
            {
                "Daily_Phone_Hours": 4.8,
                "Social_Media_Hours": 2.1,
                "Sleep_Hours": 7.3,
                "Caffeine_Intake_Cups": 1.0,
                "Weekend_Screen_Time_Hours": 6.2,
                "Age": None,
                "Gender": None,
                "Device_Type": None,
            },
            self.reference_df,
        )
        self.assertIn(result["profile"]["Gender"], self.metadata["options"]["Gender"])
        self.assertIn(
            result["profile"]["Device_Type"], self.metadata["options"]["Device_Type"]
        )


if __name__ == "__main__":
    unittest.main()
