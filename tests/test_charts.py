from __future__ import annotations

import unittest

from src.charts import (
    build_correlation_chart,
    build_feature_impact_chart,
    build_global_importance_chart,
    build_lifestyle_breakdown_chart,
)
from src.data import load_processed_dataset
from src.inference import analyze_profile


class ChartsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference_df = load_processed_dataset()
        cls.analysis = analyze_profile(
            {
                "Daily_Phone_Hours": 5.8,
                "Social_Media_Hours": 2.8,
                "Sleep_Hours": 6.8,
                "Caffeine_Intake_Cups": 2.0,
                "Weekend_Screen_Time_Hours": 7.1,
                "Age": 34,
                "Gender": "Male",
                "Device_Type": "Android",
            },
            cls.reference_df,
        )

    def test_feature_impact_chart_renders_data(self) -> None:
        figure = build_feature_impact_chart(self.analysis["local_impacts"]["stress"], "stress")
        self.assertGreater(len(figure.data), 0)

    def test_global_importance_chart_renders_data(self) -> None:
        figure = build_global_importance_chart(
            self.analysis["global_feature_importance"]["stress"], "stress"
        )
        self.assertGreater(len(figure.data), 0)
        self.assertGreater(len(figure.data[0]["x"]), 0)

    def test_lifestyle_breakdown_chart_renders_data(self) -> None:
        figure = build_lifestyle_breakdown_chart(self.analysis["lifestyle_score"])
        self.assertGreater(len(figure.data), 0)

    def test_correlation_chart_renders_data(self) -> None:
        figure = build_correlation_chart(
            self.reference_df,
            "Daily_Phone_Hours",
            "Stress_Level",
            (5.8, float(self.analysis["stress_level"])),
        )
        self.assertGreaterEqual(len(figure.data), 2)


if __name__ == "__main__":
    unittest.main()
