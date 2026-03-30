# Digital Lifestyle Analyzer

Digital Lifestyle Analyzer is a Streamlit application that predicts stress and productivity from a minimal set of digital habit inputs, compares the user with a benchmark dataset, explains model behavior, and supports what-if lifestyle simulation.

## What Is Included

- Streamlit application with interactive dashboard
- Trained RandomForest models for stress and productivity
- Preprocessor and clustering artifacts
- Scenario simulation and personalized insight generation
- Peer comparison, radar chart, lifestyle score, and correlation explorer
- Automated unit tests
- Mobile smoke-test script
- Docker deployment packaging

## Repository Layout

```text
digital_lifestyle_analyzer/
  app.py
  run_streamlit.py
  requirements.txt
  requirements-dev.txt
  Dockerfile
  .dockerignore
  CONCEPT_SPEC.md
  smart_synthetic_data.csv
  data/
  models/
  scripts/
  src/
  tests/
```

## Local Setup

1. Install Python 3.11 or newer.
2. Install runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Optional: install dev and QA dependencies:

```powershell
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
```

## Run The App

Start the Streamlit app:

```powershell
python run_streamlit.py
```

Open:

- [http://localhost:8501](http://localhost:8501)

## Public Deployment

The fastest public hosting option for this project is Streamlit Community Cloud because the app is already a native Streamlit repo and the repository is public on GitHub.

Use:

- Repository: `advaitsharma7/digital_lifestyle_analyzer`
- Branch: `master`
- App file: `app.py`

Recommended cloud settings:

- Python version: `3.11`

Official deployment docs:

- [Deploy on Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app)

After deployment, Streamlit will give you a public `*.streamlit.app` URL that anyone can open in a browser.

## Train Or Rebuild Artifacts

If you want to rebuild the models and processed dataset:

```powershell
python scripts\train_models.py
```

This regenerates:

- [stress_model.pkl](C:/code/digital_lifestyle_analyzer/models/stress_model.pkl)
- [productivity_model.pkl](C:/code/digital_lifestyle_analyzer/models/productivity_model.pkl)
- [preprocessor.pkl](C:/code/digital_lifestyle_analyzer/models/preprocessor.pkl)
- [cluster_bundle.pkl](C:/code/digital_lifestyle_analyzer/models/cluster_bundle.pkl)
- [processed_lifestyle_data.csv](C:/code/digital_lifestyle_analyzer/data/processed_lifestyle_data.csv)

## Run Tests

Run the automated test suite:

```powershell
python -m unittest discover -s tests -v
```

Current coverage focus:

- artifact metadata and model outputs
- inference behavior with and without optional inputs
- insight count and bounds
- chart generation

## Run Mobile Smoke Test

The mobile smoke test uses Playwright with an iPhone-sized viewport.

```powershell
python scripts\mobile_smoke_test.py
```

Artifacts are written to:

- `qa_artifacts/mobile-home.png`
- `qa_artifacts/mobile-smoke.json`

## Deployment With Docker

Build the image:

```powershell
docker build -t digital-lifestyle-analyzer .
```

Run the container:

```powershell
docker run --rm -p 8501:8501 digital-lifestyle-analyzer
```

Then open:

- [http://localhost:8501](http://localhost:8501)

## Deliverables Implemented

- Minimal input module with optional personalization
- Stress level prediction
- Productivity score prediction
- Peer percentile comparison
- Lifestyle balance radar chart
- Scenario simulation
- Correlation explorer
- Behavioral archetypes with clustering
- Lifestyle score and sub-score breakdown
- Explainability with both global RandomForest feature importance and personalized local effects
- Smart rule-based insights

## Notes

- Outputs are predictive estimates based on a synthetic benchmark dataset.
- The app is intended for reflection and exploration, not diagnosis or medical advice.
