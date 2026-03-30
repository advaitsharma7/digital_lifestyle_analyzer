# Digital Lifestyle Analyzer

## Detailed Concept Specification

**Document status:** Draft v1  
**Prepared from:** `Digital Lifestyle Analyzer- instructions.pdf` and `smart_synthetic_data.csv`  
**Last updated:** 2026-03-30

## 1. Product Summary

Digital Lifestyle Analyzer is a web-based AI coaching experience that helps a user understand how their current digital habits may be affecting stress and productivity. The product should feel more like a smart lifestyle advisor than a reporting dashboard.

The experience starts with a short, low-friction intake form and returns:

- A predicted stress level on a `1-5` scale
- A predicted productivity score on a `1-10` scale
- Comparison against peers in the benchmark dataset
- Interactive visual explanations of the user's behavior pattern
- A what-if simulator that shows how lifestyle changes may improve outcomes
- Personalized, plain-language insights

The product should answer three user questions:

- Where do I stand?
- Why is this happening?
- What should I do next?

## 2. Vision and Positioning

### Vision

Turn a few daily habit inputs into an intelligent coaching session that helps users reflect on their routines and explore healthier digital behavior patterns.

### Positioning Statement

For people who want a fast understanding of how screen time and daily habits affect their well-being, Digital Lifestyle Analyzer is an interactive AI lifestyle coach that predicts outcomes, benchmarks habits, and suggests improvements, unlike a static dashboard that only reports raw metrics.

### Product Personality

- Supportive, intelligent, and non-judgmental
- Visual and interactive
- Practical rather than academic
- Light enough for casual use, but credible enough to feel data-driven

## 3. Goals

### Primary Goals

- Deliver meaningful predictions from a minimal input set
- Keep the intake flow under 10 seconds
- Make simulation the centerpiece of the product experience
- Explain predictions in simple language
- Encourage action through concrete insights instead of passive charts

### Secondary Goals

- Showcase multiple analytics modes in one cohesive experience
- Make results easy to understand on desktop and mobile
- Provide a clean and polished demo-ready product

### Non-Goals for v1

- Medical diagnosis or mental health treatment guidance
- Real-time device tracking or integration with actual phone telemetry
- User accounts, history, or long-term progress tracking
- Demographic segmentation beyond simple optional fields
- Deep causal claims about behavior and health

## 4. Target Users

### Primary User

A student or working professional who wants a quick read on whether their phone, social media, sleep, and caffeine habits are helping or hurting their daily performance.

### User Motivations

- Understand whether current habits look healthy or imbalanced
- See how they compare with others
- Test small habit changes before making them
- Receive simple suggestions without filling out a long survey

### Jobs To Be Done

- When I feel my routine is off, help me quickly understand why
- When I want to improve my habits, show me the most impactful changes
- When I enter a few numbers, give me insights that feel personal and actionable

## 5. Source Inputs and Dataset

### Source Dataset

The benchmark dataset is `smart_synthetic_data.csv` with `50,000` records and the following columns:

| Column | Type | Notes |
| --- | --- | --- |
| `User_ID` | string | Identifier only; not used for modeling |
| `Age` | numeric | Optional user input |
| `Gender` | categorical | Optional user input |
| `Occupation` | categorical | Present in dataset; not collected in v1 intake |
| `Device_Type` | categorical | Optional user input |
| `Daily_Phone_Hours` | numeric | Required user input |
| `Social_Media_Hours` | numeric | Required user input |
| `Work_Productivity_Score` | numeric | Regression target |
| `Sleep_Hours` | numeric | Required user input |
| `Stress_Level` | numeric | Classification target |
| `App_Usage_Count` | numeric | Auto-generated derived feature |
| `Caffeine_Intake_Cups` | numeric | Required user input |
| `Weekend_Screen_Time_Hours` | numeric | Required user input |

### Product Decision on Dataset Alignment

To keep training and inference consistent, the live model should only depend on fields available in the app at prediction time.

That means:

- Exclude `User_ID` from all modeling
- Exclude `Occupation` from v1 models because it is not collected from the user
- Include `Age`, `Gender`, and `Device_Type` as optional features with imputation
- Generate `App_Usage_Count` from entered screen-time behavior at inference time

### UI Label to Data Column Mapping

| UI Label | Dataset Column |
| --- | --- |
| Daily Phone Hours | `Daily_Phone_Hours` |
| Social Media Hours | `Social_Media_Hours` |
| Sleep Hours | `Sleep_Hours` |
| Caffeine Intake | `Caffeine_Intake_Cups` |
| Weekend Screen Time | `Weekend_Screen_Time_Hours` |
| Age | `Age` |
| Gender | `Gender` |
| Device Type | `Device_Type` |
| Productivity Score | `Work_Productivity_Score` |
| Stress Level | `Stress_Level` |

## 6. User Input Module

### Objective

Capture enough information to generate useful predictions while keeping the experience extremely fast.

### Required Inputs

| Field | Control | Range | Default |
| --- | --- | --- | --- |
| Daily Phone Hours | Slider | `0-12` | `5.0` |
| Social Media Hours | Slider | `0-10` | `2.5` |
| Sleep Hours | Slider | `0-10` | `7.0` |
| Caffeine Intake | Slider | `0-8 cups` | `2` |
| Weekend Screen Time | Slider | `0-15` | `7.0` |

### Optional Inputs

These appear in a collapsed section labeled `Personalize results`.

| Field | Control | Values | Default Behavior |
| --- | --- | --- | --- |
| Age | Slider or number stepper | `13-80` | Impute dataset average if omitted |
| Gender | Dropdown | Dataset-supported values | Impute most common value if omitted |
| Device Type | Dropdown | Dataset-supported values | Impute most common value if omitted |

### UX Requirements

- The intake area must be visible above the fold on common laptop screens
- Required inputs must be obvious and easy to manipulate
- Users should complete the form in under 10 seconds
- Inputs should update predictions via a clear `Analyze My Lifestyle` action
- Optional fields should never block submission
- Validation must be inline and non-disruptive

### Smart Logic

- `App_Usage_Count` is auto-generated from entered screen-time values
- Missing optional fields are filled using dataset averages or modes
- The app should show a subtle note that optional inputs improve personalization

### Proposed Derived Feature Logic

For v1, `App_Usage_Count` should be estimated deterministically from screen behavior so the value is stable and explainable. A recommended formula is:

`App_Usage_Count = round((Daily_Phone_Hours * 10) + (Social_Media_Hours * 4) + (Weekend_Screen_Time_Hours * 2))`

This can later be replaced by a learned estimator if needed.

## 7. Prediction Engine

### Core Outputs

- `Stress Level` as a classification output from `1` to `5`
- `Productivity Score` as a regression output from `1` to `10`

### Model Requirements

| Model | Type | Algorithm | Target | Metric |
| --- | --- | --- | --- | --- |
| Stress Model | Classification | RandomForestClassifier | `Stress_Level` | Accuracy |
| Productivity Model | Regression | RandomForestRegressor | `Work_Productivity_Score` | RMSE |

### Train/Test Strategy

- Use an `80/20` train-test split
- Fix a random seed for reproducibility
- Store evaluation results during training for display in documentation or developer logs

### Feature Set for Inference

- `Age`
- `Gender`
- `Device_Type`
- `Daily_Phone_Hours`
- `Social_Media_Hours`
- `Sleep_Hours`
- `Caffeine_Intake_Cups`
- `Weekend_Screen_Time_Hours`
- `App_Usage_Count`

### Preprocessing Requirements

- Numeric features should be passed through with imputation as needed
- Categorical features should be encoded through a reproducible preprocessing pipeline
- The same preprocessor must be used during training and inference

### Output Artifacts

- `stress_model.pkl`
- `productivity_model.pkl`
- `preprocessor.pkl`

### Prediction Response Contract

Each analysis run should produce:

- Predicted stress class
- Predicted productivity score
- Confidence-friendly display values formatted for UI
- Feature-importance data for both models
- Cluster assignment
- Lifestyle score
- Generated insight list

## 8. Dashboard Information Architecture

The dashboard should feel like one continuous coaching session. Content should be arranged in cards with limited scrolling and a strong top-to-bottom story.

### Recommended Layout Order

1. Input panel
2. Headline prediction cards
3. Explain-why section
4. Peer and balance visuals
5. Scenario simulation
6. Correlation explorer
7. Archetype and lifestyle score
8. Personalized insights and next steps

### Design Principles

- Minimal scrolling
- Card-based layout
- Responsive across desktop and mobile
- Visually distinct primary metrics
- Plain-language labels over technical jargon
- Immediate chart readability without requiring tutorials

## 9. Functional Modules

## 9.1 Personal Metrics Overview

### Purpose

Provide the fastest possible answer to the user's current state.

### UI Components

- Stress gauge chart
- Productivity gauge chart
- Short headline summary sentence

### Example Headline

`You appear moderately stressed and below your productivity potential today.`

### Acceptance Criteria

- Both gauges render after analysis
- Stress is displayed on a `1-5` scale
- Productivity is displayed on a `1-10` scale
- Each gauge uses color coding that is easy to understand

## 9.2 Peer Comparison Module

### Purpose

Show how the user compares with the benchmark population.

### Metrics Included

- Daily Phone Hours
- Social Media Hours
- Sleep Hours

### Outputs

- Percentile rank for each metric
- Text interpretation such as `higher than 72% of peers`

### Recommended Visualization

- Horizontal percentile bars or compact comparison cards

### Acceptance Criteria

- Percentiles are computed against `smart_synthetic_data.csv`
- The app explains whether a higher or lower percentile is favorable
- The module remains understandable even if the user skips optional fields

## 9.3 Lifestyle Balance Radar Chart

### Purpose

Help users see their overall balance rather than isolated metrics.

### Axes

- Sleep
- Screen Time
- Social Media
- Stress
- Productivity

### Rules

- Normalize all displayed values to a `0-1` range
- Overlay `User` vs `Dataset Average`
- Metrics where lower is better should be inverted before plotting if needed so that larger radar area implies better balance

### Acceptance Criteria

- User polygon and dataset average polygon are both visible
- Axis labels are readable on mobile
- The chart updates after scenario simulation

## 9.4 Scenario Simulation

### Purpose

This is the signature feature. It lets users test how changes in behavior may affect outcomes.

### Adjustable Inputs

- Sleep Hours
- Social Media Hours
- Daily Phone Hours

### Outputs

- Updated stress prediction
- Updated productivity prediction
- Before vs After comparison chart
- Impact summary text

### Example Output

`If you increase sleep by 1.5 hours and reduce social media by 1 hour, your predicted stress drops from 4 to 3 and productivity rises from 5.9 to 6.8.`

### UX Requirements

- Original values remain visible
- Simulated values update interactively
- The contrast between current state and projected state is visually obvious
- Changes should feel instantaneous or near-instantaneous

### Acceptance Criteria

- Adjusting simulation inputs re-runs the prediction pipeline
- `App_Usage_Count` is recalculated during simulation
- Before and after values are both retained until the user resets
- Impact text describes direction and magnitude in plain language

## 9.5 Correlation Explorer

### Purpose

Let users explore broader relationships in the dataset.

### Controls

- X-axis variable selector
- Y-axis variable selector

### Visualization

- Interactive scatter plot using Plotly
- Regression trend line
- Tooltip with underlying values

### Recommended Variables

- Daily Phone Hours
- Social Media Hours
- Sleep Hours
- Caffeine Intake
- Weekend Screen Time
- Stress Level
- Productivity Score
- App Usage Count

### Acceptance Criteria

- Any valid variable pair can be selected
- The trend line updates after selection changes
- The chart remains performant on the full dataset

## 9.6 Behavioral Archetypes

### Purpose

Translate analytics into relatable behavioral patterns.

### Model

- `KMeans` clustering with `k=3`

### Example Labels

- High Stress Users
- Balanced Users
- Low Activity Users

### Product Decision

Cluster names and descriptions should be assigned after inspecting cluster centers. Labels must reflect real cluster characteristics instead of being hard-coded blindly.

### Output

- Cluster label
- Short description
- A sentence describing why the user belongs to that archetype

### Acceptance Criteria

- Each user is assigned exactly one cluster
- The label is human-readable
- The description references the cluster's dominant traits

## 9.7 Lifestyle Score

### Purpose

Provide a simple gamified summary score that rewards healthier balance.

### Score Definition

The final score is out of `100` and combines:

- Sleep: `30%`
- Screen Time: `25%`
- Stress: `25%`
- Productivity: `20%`

### Recommended Scoring Logic

To keep the score transparent, each component should be normalized to `0-100` before weighting.

Recommended v1 normalization:

- Sleep score: full credit around `7.5-8.5` hours, gradually lower outside that range
- Screen Time score: lower combined phone and weekend screen time yields a higher score
- Stress score: lower predicted stress yields a higher score
- Productivity score: higher predicted productivity yields a higher score

### Display

- Large score value
- Progress bar
- Component breakdown

### Acceptance Criteria

- Score recalculates after simulation
- Component weights sum to `100%`
- Users can see which component helps or hurts the total most

## 9.8 Explainable AI

### Purpose

Explain why the model made its prediction in terms the user can understand.

### Method

Use RandomForest feature importance as the baseline explainability method in v1.

### Outputs

- Plain-language explanation snippet
- Horizontal bar chart of top contributing features

### Example Explanation

`Your predicted stress is high mainly because your phone usage and social media hours are elevated while sleep is below the healthy range.`

### Product Decision

RandomForest global feature importance is acceptable for v1, but UI copy should avoid implying exact individualized causality. Language should use phrases like `main drivers`, `strongly associated`, or `likely influencing`.

### Acceptance Criteria

- The app shows the top drivers for stress
- The app shows the top drivers for productivity
- Text explanation and chart are consistent with one another

## 9.9 Smart Insights Engine

### Purpose

Translate metrics into concrete advice.

### Insight Generation

Use a rule-based system in v1 to generate `3-5` personalized insights from user inputs and model outputs.

### Example Rules

- `Sleep_Hours < 6` -> low sleep likely increases stress
- `Social_Media_Hours > 4` -> high social use may reduce focus
- `Caffeine_Intake_Cups >= 4` and `Sleep_Hours < 6` -> elevated stress risk pattern
- `Daily_Phone_Hours > 8` -> digital overload risk
- `Stress_Level >= 4` and `Productivity_Score <= 5` -> burnout-style pattern

### Output Style

- Short
- Specific
- Action-oriented
- Non-judgmental

### Acceptance Criteria

- Between `3` and `5` insights are shown
- Insights do not contradict the visible metrics
- At least one insight highlights a positive or neutral pattern when applicable

## 10. End-to-End User Flow

1. User lands on the app and sees a short intake form.
2. User enters five required values and optionally personalizes with demographic/device info.
3. User submits the form.
4. The app imputes optional fields if needed and derives `App_Usage_Count`.
5. The app runs both models and computes peer percentiles, cluster assignment, radar values, and lifestyle score.
6. The dashboard renders prediction cards first.
7. The user explores peer benchmarks and chart modules.
8. The user changes simulation controls and sees updated predictions.
9. The app closes with plain-language explanations and suggested next steps.

## 11. UX and Visual Design Requirements

### Framework and Libraries

- Framework: `Streamlit`
- Visualization library: `Plotly`

### Layout Requirements

- Clean dashboard layout
- Card-based sections
- Minimal scrolling
- Responsive design

### UX Guidance

- The page should open with a clear value proposition in one sentence
- The app should not overwhelm the user with all visuals at once
- The most important feedback should appear above the fold
- Charts should use consistent color semantics across modules
- Tooltips and helper text should explain ambiguous metrics

### Recommended Color Semantics

- Green: balanced or favorable
- Amber: caution
- Red: elevated risk or poor balance
- Blue or gray: neutral comparison and benchmark data

## 12. Technical Architecture

### Recommended App Structure

| Layer | Responsibility |
| --- | --- |
| UI layer | Streamlit layout, widgets, cards, and copy |
| Inference layer | Load models, preprocess inputs, produce predictions |
| Analytics layer | Percentiles, clustering, score computation, simulation |
| Insight layer | Rule evaluation and explanation text |
| Visualization layer | Plotly charts |
| Training pipeline | Build and export models and preprocessing artifacts |

### Recommended Repository Structure

```text
digital_lifestyle_analyzer/
  app.py
  data/
    smart_synthetic_data.csv
    processed_data.csv
  models/
    stress_model.pkl
    productivity_model.pkl
    preprocessor.pkl
  src/
    config.py
    preprocessing.py
    training.py
    inference.py
    scoring.py
    clustering.py
    insights.py
    charts.py
    utils.py
  notebooks/
  CONCEPT_SPEC.md
```

### State Management

In Streamlit, session state should preserve:

- Current input values
- Most recent prediction result
- Original scenario baseline
- Current simulation overrides

## 13. Data Processing and Modeling Details

### Data Preparation

- Drop `User_ID`
- Exclude `Occupation` from v1 inference model features
- Separate targets:
  - `Stress_Level`
  - `Work_Productivity_Score`
- Build a reusable preprocessing pipeline for numeric and categorical inputs

### Clustering Inputs

Recommended clustering features:

- `Daily_Phone_Hours`
- `Social_Media_Hours`
- `Sleep_Hours`
- `Stress_Level`
- `Work_Productivity_Score`
- `Weekend_Screen_Time_Hours`
- `Caffeine_Intake_Cups`

These should be standardized before fitting `KMeans`.

### Benchmark Calculations

Peer comparison percentiles should be computed against the full dataset distribution for each metric.

### Normalization Rules

All visual normalization logic should be centralized so radar charts, score calculations, and comparisons remain consistent across modules.

## 14. Performance and Quality Requirements

### Performance

- Initial app load should feel lightweight
- Prediction after submit should complete within about `2 seconds` on a local machine
- Scenario updates should complete within about `500 ms` where feasible
- Correlation chart interactions should remain smooth on the full dataset

### Quality

- Inputs must be validated before inference
- Model artifacts must load without retraining during app startup
- Numeric outputs must be rounded consistently for display
- Errors should fail gracefully with user-friendly messaging

### Reliability

- If a model artifact is missing, the app should show a helpful error instead of crashing silently
- If optional fields are missing, predictions should still run

## 15. Responsible Product Framing

The product should not present itself as diagnosing mental health or measuring true productivity. Copy must frame outputs as model-based estimates from habit patterns.

### Recommended Disclaimer

`These results are predictive estimates based on lifestyle patterns in a synthetic benchmark dataset. They are intended for reflection and exploration, not diagnosis or medical advice.`

## 16. Success Metrics

### Product Success Indicators

- User can complete intake in under 10 seconds
- User receives predictions with no required manual data cleanup
- Simulation is used as part of the typical session
- Insights are understandable without developer explanation
- Dashboard remains usable on both desktop and mobile

### Demo Success Indicators

- End-to-end flow works with one click after entering inputs
- All required charts render correctly
- Output feels coherent and polished
- The app clearly feels like an advisor, not a report viewer

## 17. Deliverables

### Required Deliverables

1. Trained ML models in `.pkl` format
2. Processed dataset
3. Streamlit web application
4. Interactive dashboard with predictions, visualizations, insights, and simulation
5. Well-structured codebase

### Required Model Files

- `stress_model.pkl`
- `productivity_model.pkl`
- `preprocessor.pkl`

## 18. Acceptance Checklist

The project should be considered complete when all of the following are true:

- The app accepts the required five inputs and optional personalization fields
- Stress and productivity predictions are displayed clearly
- Peer percentile comparisons work against the provided dataset
- Radar chart compares user versus dataset average
- Scenario simulation recalculates outputs live
- Correlation explorer supports interactive variable selection
- Clustering assigns a behavioral archetype
- Lifestyle score is visible with weighted breakdown
- Explainable AI shows top contributing factors
- The app generates `3-5` personalized insights
- The interface is clean, card-based, and responsive
- The final experience answers where the user stands, why it is happening, and what to do next

## 19. Open Questions for Future Iteration

- Should the app eventually collect `Occupation` to improve personalization?
- Should peer comparison support filtered cohorts such as age band or device type?
- Should explainability move from feature importance to a more local method such as SHAP?
- Should the simulation extend to caffeine and weekend screen time?
- Should the product store historical sessions for progress tracking?

## 20. Recommended v1 Implementation Priorities

### Must Have

- Intake form
- Dual prediction models
- Prediction cards
- Peer comparison
- Radar chart
- Scenario simulation
- Smart insights

### Should Have

- Behavioral archetypes
- Correlation explorer
- Lifestyle score
- Explainable AI chart

### Nice to Have

- Downloadable report
- Saved scenarios
- Cohort filtering
- Historical trend tracking

## 21. Final Product Thesis

Digital Lifestyle Analyzer succeeds if it transforms a handful of lifestyle inputs into a session that feels insightful, interactive, and motivating. The product should not merely display numbers. It should guide the user from awareness to interpretation to action in a way that feels personal and visually engaging.
