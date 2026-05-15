from __future__ import annotations

import io
from typing import Any

import pandas as pd
from flask import Flask, Response, redirect, render_template, request, url_for

from salary_pipeline import (
    ensure_artifacts,
    get_education_mapping,
    get_form_schema,
    get_teaching_documentation,
    predict_all_models,
    predict_batch,
    predict_single,
    train_and_save,
)


app = Flask(__name__)
model, model_card = ensure_artifacts()
FORM_SCHEMA = get_form_schema()
EDUCATION_MAP = get_education_mapping()


def refresh_runtime_state() -> None:
    global model, model_card, FORM_SCHEMA, EDUCATION_MAP
    model, model_card = ensure_artifacts()
    FORM_SCHEMA = get_form_schema()
    EDUCATION_MAP = get_education_mapping()


def build_base_context() -> dict[str, Any]:
    return {
        "best_model": model_card["best_model"],
        "best_model_display_name": model_card["best_model_display_name"],
        "overall_results": model_card["results"],
    }


def _build_payload(form_data: dict[str, Any]) -> dict[str, Any]:
    education = str(form_data["education"]).strip()

    return {
        "age": int(form_data["age"]),
        "work-class": str(form_data["work-class"]).strip(),
        "work-fnl": 0,
        "education": education,
        "education-num": EDUCATION_MAP[education],
        "marital-status": str(form_data["marital-status"]).strip(),
        "position": str(form_data["position"]).strip(),
        "relationship": str(form_data["relationship"]).strip(),
        "race": str(form_data["race"]).strip(),
        "sex": str(form_data["sex"]).strip(),
        "capital-gain": int(form_data["capital-gain"]),
        "capital-loss": int(form_data["capital-loss"]),
        "hours-per-week": int(form_data["hours-per-week"]),
        "native-country": str(form_data["native-country"]).strip(),
    }


@app.route("/", methods=["GET"])
def home() -> str:
    return render_template("home.html", **build_base_context())


@app.route("/documentation", methods=["GET"])
def documentation() -> str:
    return render_template(
        "documentation.html",
        doc=get_teaching_documentation(model_card["results"]),
        **build_base_context(),
    )


@app.route("/developers", methods=["GET"])
def developers() -> str:
    return render_template("developers.html", **build_base_context())


@app.route("/test", methods=["GET"])
def test_page() -> str:
    return render_template(
        "test.html",
        form_schema=FORM_SCHEMA,
        prediction=None,
        model_predictions=None,
        model_predictions_by_name={},
        batch_preview=None,
        batch_columns=None,
        batch_error=None,
        **build_base_context(),
    )


@app.route("/predict", methods=["POST"])
def predict() -> str:
    try:
        payload = _build_payload(request.form)
        prediction = predict_single(payload)
        model_predictions = predict_all_models(payload)
        model_predictions_by_name = {
            item["model_name"]: item for item in model_predictions
        }
        return render_template(
            "test.html",
            form_schema=FORM_SCHEMA,
            prediction=prediction,
            model_predictions=model_predictions,
            model_predictions_by_name=model_predictions_by_name,
            batch_preview=None,
            batch_columns=None,
            batch_error=None,
            **build_base_context(),
        )
    except Exception as exc:
        return render_template(
            "test.html",
            form_schema=FORM_SCHEMA,
            prediction={"error": str(exc)},
            model_predictions=None,
            model_predictions_by_name={},
            batch_preview=None,
            batch_columns=None,
            batch_error=None,
            **build_base_context(),
        )


@app.route("/predict-csv", methods=["POST"])
def predict_csv() -> Response | str:
    upload = request.files.get("csv_file")
    if upload is None or upload.filename == "":
        return render_template(
            "test.html",
            form_schema=FORM_SCHEMA,
            prediction=None,
            model_predictions=None,
            model_predictions_by_name={},
            batch_preview=None,
            batch_columns=None,
            batch_error="Choose a CSV file first.",
            **build_base_context(),
        )

    try:
        frame = pd.read_csv(upload)
        predicted = predict_batch(frame)
    except Exception as exc:
        return render_template(
            "test.html",
            form_schema=FORM_SCHEMA,
            prediction=None,
            model_predictions=None,
            model_predictions_by_name={},
            batch_preview=None,
            batch_columns=None,
            batch_error=str(exc),
            **build_base_context(),
        )

    if request.form.get("download") == "1":
        buffer = io.StringIO()
        predicted.to_csv(buffer, index=False)
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    preview = predicted.head(10).fillna("").to_dict(orient="records")
    return render_template(
        "test.html",
        form_schema=FORM_SCHEMA,
        prediction=None,
        model_predictions=None,
        model_predictions_by_name={},
        batch_preview=preview,
        batch_columns=list(predicted.columns),
        batch_error=None,
        **build_base_context(),
    )


@app.route("/retrain", methods=["POST"])
def retrain() -> Response:
    train_and_save()
    refresh_runtime_state()
    return redirect(url_for("developers"))


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
