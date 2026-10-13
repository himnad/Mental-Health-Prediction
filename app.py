
from pathlib import Path
from io import BytesIO
import base64
import logging
import pickle

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from werkzeug.exceptions import BadRequestKeyError


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

FEATURES = [
    "Age",
    "Gender",
    "self_employed",
    "family_history",
    "remote_work",
    "care_options",
    "anonymity",
    "leave",
    "coworkers",
    "supervisor",
    "obs_consequence",
]


# -----------------------------
# Load ML model and transformers
# Only load pickle files you trust.
# -----------------------------

def load_pickle(filename):
    path = BASE_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open("rb") as file:
        return pickle.load(file)


model = load_pickle("model.pkl")
ct = load_pickle("ct.pkl")
le = load_pickle("le.pkl")


# -----------------------------
# Load processed dataset
# -----------------------------

def load_dataset():
    path = BASE_DIR / "processed_data.csv"

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction form
# -----------------------------

@app.get("/form")
def show_form():
    return render_template("form.html")


# -----------------------------
# Read and validate form data
# -----------------------------

def get_form_data(form):

    age = int(form["inputAge"])

    if not 1 <= age <= 120:
        raise ValueError(
            "Age must be between 1 and 120."
        )

    # Use the separate supervisor field when present.
    # Fall back to the old field for compatibility
    # until form.html is updated.
    supervisor = form.get("inputSupervisor")

    if supervisor is None:
        supervisor = form["inputColleague"]

    data = {
        "Age": age,
        "Gender": form["inputGender"],
        "self_employed": form["inputSelfEmployed"],
        "family_history": form["inputFamilyHistory"],
        "remote_work": form["inputRemoteWork"],
        "care_options": form["inputCareOptions"],
        "anonymity": form["inputAnonymity"],
        "leave": form["inputLeave"],
        "coworkers": form["inputColleague"],
        "supervisor": supervisor,
        "obs_consequence": form["inputObsConsequence"],
    }

    return pd.DataFrame([data], columns=FEATURES)


# -----------------------------
# Generate comparison graph
# -----------------------------

def generate_graph(user_data):

    dataset = load_dataset()

    required_columns = set(FEATURES + ["treatment"])

    missing_columns = required_columns - set(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(sorted(missing_columns))
        )

    yes_counts = []
    no_counts = []

    for feature in FEATURES:

        selected_value = user_data.iloc[0][feature]

        matching = dataset[
            dataset[feature] == selected_value
        ]

        yes_count = (
            matching["treatment"] == "Yes"
        ).sum()

        no_count = (
            matching["treatment"] == "No"
        ).sum()

        yes_counts.append(int(yes_count))
        no_counts.append(int(no_count))

    # Keep all 11 features separate.
    positions = np.arange(len(FEATURES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    try:
        ax.bar(
            positions - width / 2,
            yes_counts,
            width,
            label="Yes",
            color="skyblue"
        )

        ax.bar(
            positions + width / 2,
            no_counts,
            width,
            label="No",
            color="orange"
        )

        ax.set_xlabel("Survey Features")
        ax.set_ylabel("Number of Respondents")

        ax.set_title(
            "Treatment History by Matching Survey Feature"
        )

        ax.set_xticks(positions)

        ax.set_xticklabels(
            FEATURES,
            rotation=45,
            ha="right"
        )

        ax.legend()
        fig.tight_layout()

        buffer = BytesIO()

        fig.savefig(
            buffer,
            format="png",
            bbox_inches="tight"
        )

        buffer.seek(0)

        encoded_image = base64.b64encode(
            buffer.read()
        ).decode("utf-8")

        buffer.close()

        return encoded_image

    finally:
        plt.close(fig)


# -----------------------------
# Prediction route
# -----------------------------

@app.post("/submit_form")
def submit_form():

    try:
        user_data = get_form_data(request.form)

        # Preprocessing
        transformed_data = ct.transform(user_data)

        # Prediction
        prediction = model.predict(transformed_data)

        # Decode prediction
        treatment = str(
            le.inverse_transform(prediction)[0]
        )

        app.logger.info(
            "Prediction completed successfully"
        )

        # Generate comparison visualization
        image_base64 = generate_graph(user_data)

        if treatment == "Yes":
            template = "treatmentY.html"

        elif treatment == "No":
            template = "treatmentN.html"

        else:
            raise ValueError(
                "Unexpected prediction label."
            )

        return render_template(
            template,
            image_base64=image_base64,
            treatment=treatment
        )

    except (ValueError, KeyError, BadRequestKeyError) as error:

        app.logger.warning(
            "Invalid form submission: %s",
            error
        )

        return (
            "Invalid form input. Please check all "
            "fields and try again.",
            400
        )

    except Exception:

        app.logger.exception(
            "Prediction failed"
        )

        return (
            "Unable to process your request. "
            "Please try again later.",
            500
        )


# -----------------------------
# Team page
# -----------------------------

@app.route("/teampage")
def team_page():
    return "Welcome to Team page."


# -----------------------------
# Run application
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )
