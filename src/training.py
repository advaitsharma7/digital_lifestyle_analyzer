from __future__ import annotations

import json

from src.config import (
    APP_USAGE_INPUT_FEATURES,
    CLUSTER_BUNDLE_PATH,
    CLUSTER_FEATURES,
    CORRELATION_FIELDS,
    METADATA_PATH,
    MODEL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_PATH,
    PRODUCTIVITY_MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    STRESS_MODEL_PATH,
    UI_FIELD_LABELS,
)
from src.data import (
    clean_dataset,
    compute_defaults,
    compute_options,
    dataset_ranges,
    ensure_directory,
)

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, ["Gender", "Device_Type"]),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def fit_app_usage_estimator(dataset: pd.DataFrame) -> dict[str, float]:
    matrix = np.column_stack(
        [
            np.ones(len(dataset)),
            dataset[APP_USAGE_INPUT_FEATURES].to_numpy(dtype=float),
        ]
    )
    target = dataset["App_Usage_Count"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return {
        "intercept": float(coefficients[0]),
        "Daily_Phone_Hours": float(coefficients[1]),
        "Social_Media_Hours": float(coefficients[2]),
        "Weekend_Screen_Time_Hours": float(coefficients[3]),
    }


def aggregate_feature_importances(
    importances: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for feature_name, importance in zip(feature_names, importances, strict=True):
        base_name = feature_name.split("__", 1)[-1]
        matched_name = base_name
        for candidate in MODEL_FEATURES:
            if base_name == candidate or base_name.startswith(f"{candidate}_"):
                matched_name = candidate
                break
        aggregated[matched_name] = aggregated.get(matched_name, 0.0) + float(importance)
    total = sum(aggregated.values()) or 1.0
    return {
        feature: round(value / total, 6)
        for feature, value in sorted(
            aggregated.items(), key=lambda item: item[1], reverse=True
        )
    }


def describe_cluster_row(label: str, center: pd.Series) -> str:
    if label == "High Stress Users":
        return (
            f"Longer screen days ({center['Daily_Phone_Hours']:.1f}h phone use), "
            f"shorter sleep ({center['Sleep_Hours']:.1f}h), and the highest stress profile."
        )
    if label == "Balanced Users":
        return (
            f"More stable sleep ({center['Sleep_Hours']:.1f}h), lower stress "
            f"({center['Stress_Level']:.1f}), and the healthiest productivity mix."
        )
    return (
        f"Lighter overall digital intensity with {center['Daily_Phone_Hours']:.1f}h of "
        f"daily phone use and calmer stress and caffeine patterns."
    )


def build_cluster_bundle(dataset: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    cluster_frame = dataset[CLUSTER_FEATURES].copy()
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(cluster_frame)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    cluster_ids = kmeans.fit_predict(scaled_values)

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_), columns=CLUSTER_FEATURES
    )

    all_ids = centers.index.tolist()
    high_stress_id = int(centers["Stress_Level"].idxmax())

    remaining_ids = [cluster_id for cluster_id in all_ids if cluster_id != high_stress_id]
    balanced_id = int(
        max(
            remaining_ids,
            key=lambda cluster_id: (
                centers.loc[cluster_id, "Work_Productivity_Score"]
                - abs(centers.loc[cluster_id, "Stress_Level"] - 2.0)
                - abs(centers.loc[cluster_id, "Sleep_Hours"] - 7.0) * 0.3
            ),
        )
    )
    low_activity_id = int(
        next(cluster_id for cluster_id in remaining_ids if cluster_id != balanced_id)
    )

    label_map = {
        high_stress_id: "High Stress Users",
        balanced_id: "Balanced Users",
        low_activity_id: "Low Activity Users",
    }
    description_map = {
        cluster_id: describe_cluster_row(label_map[cluster_id], centers.loc[cluster_id])
        for cluster_id in all_ids
    }

    labeled_dataset = dataset.copy()
    labeled_dataset["Cluster_Id"] = cluster_ids
    labeled_dataset["Cluster_Label"] = labeled_dataset["Cluster_Id"].map(label_map)
    labeled_dataset["Cluster_Description"] = labeled_dataset["Cluster_Id"].map(
        description_map
    )

    return (
        {
            "model": kmeans,
            "scaler": scaler,
            "features": CLUSTER_FEATURES,
            "label_map": label_map,
            "description_map": description_map,
            "centers": centers.to_dict(orient="index"),
        },
        labeled_dataset,
    )


def load_metadata() -> dict[str, object]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def artifacts_exist() -> bool:
    required_paths = [
        STRESS_MODEL_PATH,
        PRODUCTIVITY_MODEL_PATH,
        PREPROCESSOR_PATH,
        CLUSTER_BUNDLE_PATH,
        METADATA_PATH,
        PROCESSED_DATA_PATH,
    ]
    return all(path.exists() for path in required_paths)


def ensure_artifacts(force: bool = False) -> dict[str, object]:
    if force or not artifacts_exist():
        return train_models()
    return load_metadata()


def train_models() -> dict[str, object]:
    ensure_directory(MODELS_DIR)
    ensure_directory(PROCESSED_DATA_PATH.parent)

    raw_dataset = clean_dataset(pd.read_csv(RAW_DATA_PATH))
    model_dataset = raw_dataset.drop(columns=["User_ID", "Occupation"]).copy()

    defaults = compute_defaults(model_dataset)
    options = compute_options(model_dataset)
    ranges = dataset_ranges(model_dataset, NUMERIC_FEATURES + CORRELATION_FIELDS)
    app_usage_estimator = fit_app_usage_estimator(model_dataset)

    features = model_dataset[MODEL_FEATURES].copy()
    stress_target = model_dataset["Stress_Level"].astype(int)
    productivity_target = model_dataset["Work_Productivity_Score"].astype(float)

    (
        x_train,
        x_test,
        stress_train,
        stress_test,
        productivity_train,
        productivity_test,
    ) = train_test_split(
        features,
        stress_target,
        productivity_target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stress_target,
    )

    preprocessor = build_preprocessor()
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    stress_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    stress_model.fit(x_train_transformed, stress_train)

    productivity_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    productivity_model.fit(x_train_transformed, productivity_train)

    stress_predictions = stress_model.predict(x_test_transformed)
    productivity_predictions = productivity_model.predict(x_test_transformed)

    transformed_feature_names = preprocessor.get_feature_names_out().tolist()
    stress_feature_importance = aggregate_feature_importances(
        stress_model.feature_importances_, transformed_feature_names
    )
    productivity_feature_importance = aggregate_feature_importances(
        productivity_model.feature_importances_, transformed_feature_names
    )

    cluster_bundle, processed_dataset = build_cluster_bundle(model_dataset)
    processed_dataset.to_csv(PROCESSED_DATA_PATH, index=False)

    artifact_dump_options = {"compress": 3}
    joblib.dump(stress_model, STRESS_MODEL_PATH, **artifact_dump_options)
    joblib.dump(productivity_model, PRODUCTIVITY_MODEL_PATH, **artifact_dump_options)
    joblib.dump(preprocessor, PREPROCESSOR_PATH, **artifact_dump_options)
    joblib.dump(cluster_bundle, CLUSTER_BUNDLE_PATH, **artifact_dump_options)

    metadata = {
        "feature_order": MODEL_FEATURES,
        "field_labels": UI_FIELD_LABELS,
        "defaults": defaults,
        "options": options,
        "ranges": ranges,
        "metrics": {
            "stress_accuracy": round(
                float(accuracy_score(stress_test, stress_predictions)), 4
            ),
            "productivity_rmse": round(
                float(
                    np.sqrt(
                        mean_squared_error(
                            productivity_test,
                            productivity_predictions,
                        )
                    )
                ),
                4,
            ),
        },
        "feature_importance": {
            "stress": stress_feature_importance,
            "productivity": productivity_feature_importance,
        },
        "app_usage_estimator": app_usage_estimator,
        "app_usage_bounds": {
            "min": int(round(model_dataset["App_Usage_Count"].min())),
            "max": int(round(model_dataset["App_Usage_Count"].max())),
        },
        "dataset_summary": {
            "rows": int(len(model_dataset)),
            "correlation_fields": CORRELATION_FIELDS,
        },
        "cluster_centers": cluster_bundle["centers"],
        "cluster_labels": cluster_bundle["label_map"],
        "cluster_descriptions": cluster_bundle["description_map"],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata
