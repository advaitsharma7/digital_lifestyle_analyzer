from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.training import train_models


if __name__ == "__main__":
    metadata = train_models()
    print("Training complete.")
    print(
        f"Stress accuracy: {metadata['metrics']['stress_accuracy']}, "
        f"Productivity RMSE: {metadata['metrics']['productivity_rmse']}"
    )
