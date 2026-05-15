from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "salary_model.joblib"
ALL_MODELS_PATH = ARTIFACTS_DIR / "salary_models.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
REPORT_PATH = ROOT / "report.txt"
NOTEBOOK_FIGURE_DIR = "report_figures"

TRAIN_PATH = ROOT / "Train.csv"
TEST_PATH = ROOT / "Test.csv"

TARGET_COLUMN = "salary"
RAW_FEATURE_COLUMNS = [
    "age",
    "work-class",
    "work-fnl",
    "education",
    "education-num",
    "marital-status",
    "position",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]
TEST_RENAME_MAP = {
    "workclass": "work-class",
    "occupation": "position",
    "fnlwgt": "work-fnl",
}
SALARY_MAP = {"<=50K": 0, ">50K": 1}
SALARY_LABELS = {0: "<=50K", 1: ">50K"}
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost Classifier",
    "lightgbm": "LightGBM Classifier",
    "mlp": "Neural Network (MLP)",
}

NUMERIC_INPUT_COLUMNS = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
CATEGORICAL_INPUT_COLUMNS = [
    "work-class",
    "marital-status",
    "position",
    "relationship",
    "race",
    "sex",
    "native-country",
]
ENGINEERED_NUMERIC_COLUMNS = [
    "capital-gain-log",
    "hours_education",
    "is_family_head",
    "is_married",
    "is_native_us",
]
MODEL_NUMERIC_COLUMNS = NUMERIC_INPUT_COLUMNS + ENGINEERED_NUMERIC_COLUMNS
MODEL_CATEGORICAL_COLUMNS = CATEGORICAL_INPUT_COLUMNS
MIN_DISPLAY_PROBABILITY = 0.001
MAX_DISPLAY_PROBABILITY = 0.999

FIELD_LABELS = {
    "age": "Age",
    "education": "Education Level",
    "work-class": "Employment Type",
    "marital-status": "Marital Status",
    "position": "Job Role",
    "relationship": "Household Relationship",
    "race": "Race",
    "sex": "Sex",
    "native-country": "Country",
    "capital-gain": "Capital Gain",
    "capital-loss": "Capital Loss",
    "hours-per-week": "Work Hours Per Week",
}

FIELD_HELP_TEXT = {
    "age": "Choose the person's age in years.",
    "education": "Pick the highest education level completed.",
    "work-class": "Pick the type of employer or work arrangement.",
    "marital-status": "Pick the current marital status.",
    "position": "Pick the closest job role.",
    "relationship": "Pick the person's relationship in the household.",
    "race": "Pick the race category used in the dataset.",
    "sex": "Pick the recorded sex value used in the dataset.",
    "native-country": "Pick the country listed for the person.",
    "capital-gain": "Pick the annual capital gain amount from the dataset scale.",
    "capital-loss": "Pick the annual capital loss amount from the dataset scale.",
    "hours-per-week": "Pick the typical number of work hours per week.",
}

WORKCLASS_LABELS = {
    "Federal-gov": "Federal government",
    "Local-gov": "Local government",
    "Never-worked": "Never worked",
    "Private": "Private company",
    "Self-emp-inc": "Self-employed (incorporated)",
    "Self-emp-not-inc": "Self-employed (not incorporated)",
    "State-gov": "State government",
    "Without-pay": "Without pay",
}
MARITAL_LABELS = {
    "Divorced": "Divorced",
    "Married-AF-spouse": "Married to Armed Forces spouse",
    "Married-civ-spouse": "Married with civilian spouse",
    "Married-spouse-absent": "Married, spouse absent",
    "Never-married": "Never married",
    "Separated": "Separated",
    "Widowed": "Widowed",
}
POSITION_LABELS = {
    "Adm-clerical": "Administrative / clerical",
    "Armed-Forces": "Armed forces",
    "Craft-repair": "Craft / repair",
    "Exec-managerial": "Executive / managerial",
    "Farming-fishing": "Farming / fishing",
    "Handlers-cleaners": "Handlers / cleaners",
    "Machine-op-inspct": "Machine operator / inspector",
    "Other-service": "Other service",
    "Priv-house-serv": "Private house service",
    "Prof-specialty": "Professional specialty",
    "Protective-serv": "Protective service",
    "Sales": "Sales",
    "Tech-support": "Technical support",
    "Transport-moving": "Transport / moving",
}
RELATIONSHIP_LABELS = {
    "Husband": "Husband",
    "Not-in-family": "Not in family",
    "Other-relative": "Other relative",
    "Own-child": "Own child",
    "Unmarried": "Unmarried",
    "Wife": "Wife",
}


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Normalizes raw notebook inputs and creates model features."""

    def __init__(self) -> None:
        self.numeric_caps_: dict[str, tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureBuilder":
        frame = self._prepare_raw_frame(X)
        for col in ["age", "hours-per-week"]:
            q1 = frame[col].quantile(0.25)
            q3 = frame[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.numeric_caps_[col] = (float(lower), float(upper))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._prepare_raw_frame(X)

        for col, (lower, upper) in self.numeric_caps_.items():
            frame[col] = frame[col].clip(lower=lower, upper=upper)

        frame["capital-gain-log"] = 0.0
        gain_mask = frame["capital-gain"] > 0
        frame.loc[gain_mask, "capital-gain-log"] = np.log(
            frame.loc[gain_mask, "capital-gain"]
        )
        frame["hours_education"] = (
            frame["hours-per-week"] / frame["education-num"].replace(0, np.nan)
        ).fillna(0.0)
        frame["hours_education"] = frame["hours_education"].round(3)
        frame["is_family_head"] = (
            (frame["marital-status"] == "Married-civ-spouse")
            & (frame["relationship"].isin(["Husband", "Wife"]))
        ).astype(int)
        frame["is_married"] = frame["marital-status"].str.startswith("Married").astype(int)
        frame["is_native_us"] = (frame["native-country"] == "United-States").astype(int)

        return frame[MODEL_NUMERIC_COLUMNS + MODEL_CATEGORICAL_COLUMNS]

    @staticmethod
    def _prepare_raw_frame(X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy().rename(columns=TEST_RENAME_MAP)

        missing = [col for col in RAW_FEATURE_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        frame = frame[RAW_FEATURE_COLUMNS].copy()

        for col in frame.select_dtypes(include="object").columns:
            frame[col] = frame[col].astype(str).str.strip()

        for col in NUMERIC_INPUT_COLUMNS + ["work-fnl"]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        if frame[NUMERIC_INPUT_COLUMNS + ["work-fnl"]].isna().any().any():
            raise ValueError("Numeric input contains invalid or missing values.")

        return frame


@dataclass
class TrainingResult:
    model_name: str
    display_name: str
    train_accuracy: float
    test_accuracy: float
    classification_report: dict[str, Any]


def build_model_candidates() -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(C=1000, max_iter=1000, random_state=42),
        "svm": LinearSVC(C=1.0, random_state=42),
        "decision_tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            criterion="entropy",
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=11, random_state=42),
    }


def build_pipeline(model: Any) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), MODEL_NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                MODEL_CATEGORICAL_COLUMNS,
            ),
        ],
        sparse_threshold=0,
    )
    return Pipeline(
        steps=[
            ("feature_builder", FeatureBuilder()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Train.csv and Test.csv must exist in the project root.")

    train_df = pd.read_csv(TRAIN_PATH).rename(columns=TEST_RENAME_MAP)
    test_df = pd.read_csv(TEST_PATH).rename(columns=TEST_RENAME_MAP)

    train_df = train_df.drop_duplicates().reset_index(drop=True)
    test_df = test_df.drop_duplicates().reset_index(drop=True)

    return train_df, test_df


def encode_target(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    mapped = normalized.map(SALARY_MAP)
    if mapped.isna().any():
        bad_values = sorted(normalized[mapped.isna()].unique().tolist())
        raise ValueError(f"Unexpected salary labels: {bad_values}")
    return mapped.astype(int)


def train_and_save() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_raw_datasets()
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = encode_target(train_df[TARGET_COLUMN])
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = encode_target(test_df[TARGET_COLUMN])

    results: list[TrainingResult] = []
    fitted_models: dict[str, Pipeline] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_score = -1.0

    for model_name, model in build_model_candidates().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        fitted_models[model_name] = pipeline

        train_predictions = pipeline.predict(X_train)
        test_predictions = pipeline.predict(X_test)
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)

        results.append(
            TrainingResult(
                model_name=model_name,
                display_name=MODEL_DISPLAY_NAMES[model_name],
                train_accuracy=train_accuracy,
                test_accuracy=test_accuracy,
                classification_report=classification_report(
                    y_test,
                    test_predictions,
                    output_dict=True,
                    zero_division=0,
                ),
            )
        )

        if test_accuracy > best_score:
            best_score = test_accuracy
            best_name = model_name
            best_pipeline = pipeline

    assert best_pipeline is not None

    joblib.dump(best_pipeline, MODEL_PATH)
    joblib.dump(fitted_models, ALL_MODELS_PATH)

    model_card = {
        "best_model": best_name,
        "best_model_display_name": MODEL_DISPLAY_NAMES[best_name],
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "results": [asdict(result) for result in results],
        "form_options": get_form_options(train_df, test_df),
    }
    METRICS_PATH.write_text(json.dumps(model_card, indent=2))
    return model_card


def load_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return train_and_save()
    return json.loads(METRICS_PATH.read_text())


def load_model() -> Pipeline:
    if not MODEL_PATH.exists():
        train_and_save()
    return joblib.load(MODEL_PATH)


def load_models() -> dict[str, Pipeline]:
    if not ALL_MODELS_PATH.exists():
        train_and_save()
    return joblib.load(ALL_MODELS_PATH)


def ensure_artifacts() -> tuple[Pipeline, dict[str, Any]]:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists() or not ALL_MODELS_PATH.exists():
        model_card = train_and_save()
    else:
        model_card = load_metrics()
    return load_model(), model_card


def get_form_options(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    if train_df is None or test_df is None:
        train_df, test_df = load_raw_datasets()

    combined = pd.concat([train_df, test_df], ignore_index=True)
    options: dict[str, list[str]] = {}

    for col in CATEGORICAL_INPUT_COLUMNS:
        values = (
            combined[col]
            .astype(str)
            .str.strip()
            .replace("?", np.nan)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
        )
        options[col] = values

    return options


def get_education_mapping(
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict[str, int]:
    if train_df is None or test_df is None:
        train_df, test_df = load_raw_datasets()

    combined = pd.concat([train_df, test_df], ignore_index=True).copy()
    combined["education"] = combined["education"].astype(str).str.strip()

    pairs = (
        combined[["education", "education-num"]]
        .drop_duplicates()
        .sort_values(["education-num", "education"])
    )
    return {row["education"]: int(row["education-num"]) for _, row in pairs.iterrows()}


def make_select_options(values: list[Any], labeler: Any) -> list[dict[str, Any]]:
    return [{"value": value, "label": labeler(value)} for value in values]


def get_form_schema() -> list[dict[str, Any]]:
    train_df, test_df = load_raw_datasets()
    combined = pd.concat([train_df, test_df], ignore_index=True).rename(columns=TEST_RENAME_MAP)
    education_map = get_education_mapping(train_df, test_df)
    category_options = get_form_options(train_df, test_df)

    age_values = list(range(int(combined["age"].min()), int(combined["age"].max()) + 1))
    hours_values = sorted(combined["hours-per-week"].astype(int).unique().tolist())
    gain_values = sorted(combined["capital-gain"].astype(int).unique().tolist())
    loss_values = sorted(combined["capital-loss"].astype(int).unique().tolist())

    schema = [
        {
            "name": "age",
            "label": FIELD_LABELS["age"],
            "help_text": FIELD_HELP_TEXT["age"],
            "options": make_select_options(age_values, lambda value: f"{value} years old"),
        },
        {
            "name": "education",
            "label": FIELD_LABELS["education"],
            "help_text": FIELD_HELP_TEXT["education"],
            "options": make_select_options(
                list(education_map.keys()),
                lambda value: value.replace("-", " "),
            ),
        },
        {
            "name": "work-class",
            "label": FIELD_LABELS["work-class"],
            "help_text": FIELD_HELP_TEXT["work-class"],
            "options": make_select_options(
                category_options["work-class"],
                lambda value: WORKCLASS_LABELS.get(value, value),
            ),
        },
        {
            "name": "marital-status",
            "label": FIELD_LABELS["marital-status"],
            "help_text": FIELD_HELP_TEXT["marital-status"],
            "options": make_select_options(
                category_options["marital-status"],
                lambda value: MARITAL_LABELS.get(value, value),
            ),
        },
        {
            "name": "position",
            "label": FIELD_LABELS["position"],
            "help_text": FIELD_HELP_TEXT["position"],
            "options": make_select_options(
                [value for value in category_options["position"] if value != "?"],
                lambda value: POSITION_LABELS.get(value, value),
            ),
        },
        {
            "name": "relationship",
            "label": FIELD_LABELS["relationship"],
            "help_text": FIELD_HELP_TEXT["relationship"],
            "options": make_select_options(
                category_options["relationship"],
                lambda value: RELATIONSHIP_LABELS.get(value, value),
            ),
        },
        {
            "name": "race",
            "label": FIELD_LABELS["race"],
            "help_text": FIELD_HELP_TEXT["race"],
            "options": make_select_options(category_options["race"], lambda value: value),
        },
        {
            "name": "sex",
            "label": FIELD_LABELS["sex"],
            "help_text": FIELD_HELP_TEXT["sex"],
            "options": make_select_options(category_options["sex"], lambda value: value),
        },
        {
            "name": "native-country",
            "label": FIELD_LABELS["native-country"],
            "help_text": FIELD_HELP_TEXT["native-country"],
            "options": make_select_options(category_options["native-country"], lambda value: value),
        },
        {
            "name": "capital-gain",
            "label": FIELD_LABELS["capital-gain"],
            "help_text": FIELD_HELP_TEXT["capital-gain"],
            "options": make_select_options(
                gain_values,
                lambda value: "No capital gain" if value == 0 else f"${value:,}",
            ),
        },
        {
            "name": "capital-loss",
            "label": FIELD_LABELS["capital-loss"],
            "help_text": FIELD_HELP_TEXT["capital-loss"],
            "options": make_select_options(
                loss_values,
                lambda value: "No capital loss" if value == 0 else f"${value:,}",
            ),
        },
        {
            "name": "hours-per-week",
            "label": FIELD_LABELS["hours-per-week"],
            "help_text": FIELD_HELP_TEXT["hours-per-week"],
            "options": make_select_options(hours_values, lambda value: f"{value} hours per week"),
        },
    ]
    return schema


def get_report_text() -> str:
    if not REPORT_PATH.exists():
        return ""
    return REPORT_PATH.read_text(encoding="utf-8-sig").strip()


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped == "________________" or stripped == "---|---|"


def _is_section_heading(line: str) -> bool:
    return bool(re.match(r"^\d+\.\s+.+", line.strip()))


def _is_subsection_heading(line: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\s+.+", line.strip()))


def _parse_markdown_table(lines: list[str], start: int) -> tuple[dict[str, Any] | None, int]:
    table_lines: list[str] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        table_lines.append(lines[index].strip())
        index += 1

    if len(table_lines) < 2:
        return None, start

    rows: list[list[str]] = []
    for line in table_lines:
        if set(line.replace("|", "").strip()) == {"-"}:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        return None, start

    return {
        "type": "table",
        "headers": rows[0],
        "rows": rows[1:],
    }, index


def _looks_like_table_chunk(chunk: list[str]) -> bool:
    if len(chunk) < 4:
        return False
    if any(
        line.startswith("* ")
        or re.match(r"^\d+\.\s+", line)
        or line.endswith(":")
        or "." in line
        for line in chunk
    ):
        return False
    return True


def _parse_plain_table(chunk: list[str]) -> dict[str, Any] | None:
    for column_count in (3, 2):
        if len(chunk) >= column_count * 2 and len(chunk) % column_count == 0:
            headers = chunk[:column_count]
            rows = [
                chunk[index:index + column_count]
                for index in range(column_count, len(chunk), column_count)
            ]
            if all(len(row) == column_count for row in rows):
                return {
                    "type": "table",
                    "headers": headers,
                    "rows": rows,
                }
    return None


def _parse_section_blocks(content: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in content.splitlines()]
    blocks: list[dict[str, Any]] = []
    index = 0

    while index < len(lines):
        line = lines[index]

        if _is_separator_line(line):
            index += 1
            continue

        if line.startswith("|"):
            table_block, next_index = _parse_markdown_table(lines, index)
            if table_block is not None:
                blocks.append(table_block)
                index = next_index
                continue

        if _is_subsection_heading(line):
            blocks.append({"type": "subheading", "text": line})
            index += 1
            continue

        if line.startswith("* "):
            items: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("* "):
                items.append(lines[index].strip()[2:].strip())
                index += 1
            blocks.append({"type": "bullets", "items": items})
            continue

        if re.match(r"^\d+\.\s+", line):
            items: list[str] = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[index].strip()))
                index += 1
            blocks.append({"type": "numbered", "items": items})
            continue

        if index + 1 < len(lines) and (
            lines[index + 1].startswith("* ")
            or re.match(r"^\d+\.\s+", lines[index + 1])
        ):
            blocks.append({"type": "label", "text": line})
            index += 1
            continue

        chunk: list[str] = []
        probe = index
        while probe < len(lines):
            candidate = lines[probe]
            if (
                _is_separator_line(candidate)
                or candidate.startswith("* ")
                or candidate.startswith("|")
                or _is_subsection_heading(candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            chunk.append(candidate)
            probe += 1

        chunk = [item for item in chunk if item]
        if _looks_like_table_chunk(chunk):
            table_block = _parse_plain_table(chunk)
            if table_block is not None:
                blocks.append(table_block)
                index = probe
                continue

        paragraph_lines: list[str] = []
        while index < len(lines):
            candidate = lines[index]
            if (
                _is_separator_line(candidate)
                or candidate.startswith("* ")
                or candidate.startswith("|")
                or _is_subsection_heading(candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            if candidate:
                paragraph_lines.append(candidate)
            index += 1

        if paragraph_lines:
            blocks.append({"type": "paragraph", "text": " ".join(paragraph_lines)})
        else:
            index += 1

    return blocks


def get_report_document() -> dict[str, Any]:
    text = get_report_text()
    if not text:
        return {"title": "Project Report", "toc": [], "sections": []}

    lines = text.splitlines()
    title = lines[0].strip() if lines else "Project Report"

    all_heading_matches = list(re.finditer(r"(?m)^(?P<num>\d+)\.\s+(?P<title>.+)$", text))
    intro_positions = [
        match.start()
        for match in all_heading_matches
        if match.group("num") == "1" and match.group("title").strip() == "Introduction"
    ]
    actual_start = intro_positions[1] if len(intro_positions) >= 2 else 0

    toc_source = text[:actual_start] if actual_start else text
    toc_matches = list(re.finditer(r"(?m)^(?P<num>\d+)\.\s+(?P<title>.+)$", toc_source))

    section_matches: list[tuple[str, str, re.Match[str]]] = []
    if toc_matches and actual_start:
        cursor = actual_start
        for toc_match in toc_matches:
            section_number = toc_match.group("num")
            section_title = toc_match.group("title").strip()
            pattern = re.compile(
                rf"(?m)^{re.escape(section_number)}\.\s+{re.escape(section_title)}$"
            )
            match = pattern.search(text, cursor)
            if match:
                section_matches.append((section_number, section_title, match))
                cursor = match.end()
    else:
        section_matches = [
            (match.group("num"), match.group("title").strip(), match)
            for match in all_heading_matches
        ]

    sections: list[dict[str, Any]] = []
    toc: list[dict[str, Any]] = []

    for idx, (section_number, section_title, match) in enumerate(section_matches):
        start = match.end()
        end = section_matches[idx + 1][2].start() if idx + 1 < len(section_matches) else len(text)
        content = text[start:end].strip()

        section_id = f"section-{section_number}"
        toc.append({
            "id": section_id,
            "number": section_number,
            "title": section_title,
        })
        sections.append({
            "id": section_id,
            "number": section_number,
            "title": section_title,
            "blocks": _parse_section_blocks(content),
        })

    return {
        "title": title,
        "toc": toc,
        "sections": sections,
    }


def get_teaching_documentation(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    ranked_models = sorted(
        model_results,
        key=lambda item: item["test_accuracy"],
        reverse=True,
    )

    dataset_features = [
        ("age", "Employee age and rough seniority signal."),
        ("work-class", "Type of employer such as private sector or government."),
        ("education", "Highest education level completed."),
        ("education-num", "Numeric encoding of education level."),
        ("marital-status", "Family status, which often correlates with life stage."),
        ("position / occupation", "Job family and responsibility level."),
        ("relationship", "Household role such as husband, wife, own child, or not in family."),
        ("race", "Demographic feature present in the dataset."),
        ("sex", "Demographic feature present in the dataset."),
        ("capital-gain / capital-loss", "Investment-related financial signals."),
        ("hours-per-week", "Workload intensity and possible responsibility signal."),
        ("native-country", "Country field used as a categorical feature."),
        ("salary", "Target class: income at or below 50K vs above 50K."),
    ]

    preprocessing_rows = [
        ("Rename inconsistent columns", "Make train and test schemas match before modeling."),
        ("Drop weak columns", "Remove `education` duplication and ignore `fnlwgt` as a direct predictor."),
        ("Handle duplicates", "Reduce memorization risk and improve data quality."),
        ("Treat outliers", "Cap extreme values in age and weekly hours using IQR logic."),
        ("Engineer features", "Create `capital-gain-log`, `hours_education`, and family-status flags."),
        ("Encode categories", "Convert categorical features into machine-usable numeric representations."),
        ("Scale numeric features", "Help sensitive models like Logistic Regression, SVM, and MLP train more stably."),
    ]

    relationship_figures = [
        {
            "title": "Numerical Outliers Before Treatment",
            "file": f"{NOTEBOOK_FIGURE_DIR}/eda_boxplots_before.png",
            "takeaway": "Before preprocessing, age and hours-per-week show visible spread and potential extreme values.",
        },
        {
            "title": "Numerical Outliers After Treatment",
            "file": f"{NOTEBOOK_FIGURE_DIR}/eda_boxplots_after.png",
            "takeaway": "After capping extremes, the distributions are easier for models to learn from.",
        },
        {
            "title": "Occupation vs Salary",
            "file": f"{NOTEBOOK_FIGURE_DIR}/eda_position_salary.png",
            "takeaway": "Managerial and professional roles appear much more often in the higher-income class.",
        },
        {
            "title": "Work Class vs Salary",
            "file": f"{NOTEBOOK_FIGURE_DIR}/eda_workclass_salary.png",
            "takeaway": "Employment type matters. Private and self-employment categories dominate the dataset and show different class balances.",
        },
        {
            "title": "Sex vs Salary",
            "file": f"{NOTEBOOK_FIGURE_DIR}/eda_sex_salary.png",
            "takeaway": "The salary split differs by sex in this dataset, so the feature carries predictive information.",
        },
        {
            "title": "Education Number vs Hours Worked",
            "file": f"{NOTEBOOK_FIGURE_DIR}/eda_hours_education.png",
            "takeaway": "Workload and education interact, which motivated the engineered `hours_education` feature.",
        },
    ]

    confusion_figures = [
        ("Logistic Regression", f"{NOTEBOOK_FIGURE_DIR}/cm_logistic_regression.png"),
        ("SVM", f"{NOTEBOOK_FIGURE_DIR}/cm_svm.png"),
        ("Decision Tree", f"{NOTEBOOK_FIGURE_DIR}/cm_decision_tree.png"),
        ("Random Forest", f"{NOTEBOOK_FIGURE_DIR}/cm_random_forest.png"),
        ("XGBoost Classifier", f"{NOTEBOOK_FIGURE_DIR}/cm_xgboost.png"),
        ("LightGBM Classifier", f"{NOTEBOOK_FIGURE_DIR}/cm_lightgbm.png"),
        ("Neural Network (MLP)", f"{NOTEBOOK_FIGURE_DIR}/cm_mlp.png"),
    ]
    confusion_panels = []
    by_name = {item["display_name"]: item for item in ranked_models}
    for display_name, file_name in confusion_figures:
        model = by_name.get(display_name)
        if model is None:
            continue
        confusion_panels.append(
            {
                "title": display_name,
                "file": file_name,
                "score": model["test_accuracy"],
            }
        )

    sidebar_topics = [
        {
            "title": "Problem Framing",
            "note": "Define salary prediction as a binary classification task.",
        },
        {
            "title": "Dataset Reading",
            "note": "Examine the target variable, feature types, and initial data risks before implementation.",
        },
        {
            "title": "Data Cleaning",
            "note": "Clarify the effect of duplicates, unknown values, and outliers on model quality.",
        },
        {
            "title": "EDA and Relationships",
            "note": "Use visual evidence to identify variables associated with salary prediction.",
        },
        {
            "title": "Feature Engineering",
            "note": "Construct informative features from the original variables.",
        },
        {
            "title": "Model Comparison",
            "note": "Compare linear, tree-based, boosting, and neural approaches under the same task.",
        },
        {
            "title": "Evaluation Logic",
            "note": "Interpret accuracy together with confusion matrices and class-level behavior.",
        },
        {
            "title": "Academic Conclusion",
            "note": "Summarize the principal lessons and project-level findings.",
        },
    ]

    return {
        "title": "Employee Salary Prediction",
        "subtitle": "A formal walkthrough of the notebook, rewritten to explain the machine learning reasoning behind each stage of the project.",
        "sidebar_topics": sidebar_topics,
        "dataset_features": dataset_features,
        "preprocessing_rows": preprocessing_rows,
        "relationship_figures": relationship_figures,
        "ranked_models": ranked_models,
        "confusion_panels": confusion_panels,
        "best_model_name": ranked_models[0]["display_name"] if ranked_models else "",
        "best_model_score": ranked_models[0]["test_accuracy"] if ranked_models else 0.0,
        "teacher_takeaways": [
            "A successful machine learning project begins by translating the application objective into a clearly defined predictive task.",
            "Exploratory data analysis should guide preprocessing and feature engineering rather than serve as a purely visual addition.",
            "Model selection must be evidence-based and should rely on measured validation performance rather than preference for a specific algorithm.",
            "Confusion matrices remain essential because they reveal the type of classification errors that overall accuracy alone may hide.",
        ],
        "future_improvements": [
            "Apply cross-validation instead of relying on a single train/test split in order to obtain more stable performance estimates.",
            "Investigate class-imbalance handling strategies such as class weighting or SMOTE-style resampling.",
            "Incorporate explainability methods such as SHAP to clarify the reasoning of the strongest boosting models.",
            "Perform more systematic hyperparameter optimization through grid search or Bayesian optimization methods.",
        ],
    }


def _sanitize_display_probabilities(probabilities: list[float] | None) -> list[float] | None:
    if probabilities is None:
        return None

    cleaned = [float(value) for value in probabilities]
    if len(cleaned) != 2:
        return cleaned

    positive = min(max(cleaned[1], MIN_DISPLAY_PROBABILITY), MAX_DISPLAY_PROBABILITY)
    negative = 1.0 - positive
    return [negative, positive]


def _build_prediction_payload(prediction: int, probabilities: list[float] | None) -> dict[str, Any]:
    display_probabilities = _sanitize_display_probabilities(probabilities)
    confidence = None
    if display_probabilities is not None:
        confidence = max(display_probabilities)

    return {
        "prediction_code": prediction,
        "prediction_label": SALARY_LABELS[prediction],
        "probabilities": display_probabilities,
        "confidence": confidence,
    }


def predict_single(payload: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    frame = pd.DataFrame([payload])
    prediction = int(model.predict(frame)[0])

    probabilities: list[float] | None = None
    if hasattr(model, "predict_proba"):
        probabilities = [float(value) for value in model.predict_proba(frame)[0].tolist()]

    return _build_prediction_payload(prediction, probabilities)


def predict_all_models(payload: dict[str, Any]) -> list[dict[str, Any]]:
    frame = pd.DataFrame([payload])
    outputs: list[dict[str, Any]] = []

    for model_name, model in load_models().items():
        prediction = int(model.predict(frame)[0])
        probabilities: list[float] | None = None
        if hasattr(model, "predict_proba"):
            probabilities = [float(value) for value in model.predict_proba(frame)[0].tolist()]

        output = _build_prediction_payload(prediction, probabilities)
        output["model_name"] = model_name
        output["display_name"] = MODEL_DISPLAY_NAMES[model_name]
        outputs.append(output)

    return outputs


def predict_batch(frame: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    prepared = frame.copy().rename(columns=TEST_RENAME_MAP)
    predictions = model.predict(prepared)
    output = frame.copy()
    output["predicted_salary"] = [SALARY_LABELS[int(value)] for value in predictions]
    return output
