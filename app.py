
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------------------
# IMPORTANT: Explicitly import Interpret before unpickling the EBM.
# This prevents Streamlit/joblib from failing to resolve the package.
# ---------------------------------------------------------------------
try:
    import interpret
    from interpret.glassbox import ExplainableBoostingClassifier
    INTERPRET_IMPORT_ERROR = None
except Exception as exc:
    interpret = None
    ExplainableBoostingClassifier = None
    INTERPRET_IMPORT_ERROR = exc

import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="MCI → AD Conversion Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "Results"

# ---------------------------------------------------------------------
# Project feature definition
# These are the 9 primary features used in the research model.
# ---------------------------------------------------------------------
NUMERIC_FEATURES = [
    "AGE",
    "PTEDUCAT",
    "CDGLOBAL",
    "CDRSB",
    "MMSCORE",
]

CATEGORICAL_FEATURES = [
    "PTGENDER",
    "PTETHCAT",
    "PTRACCAT",
    "APOE",
]

FEATURE_ORDER = [
    "AGE",
    "PTGENDER",
    "PTEDUCAT",
    "PTETHCAT",
    "PTRACCAT",
    "APOE",
    "CDGLOBAL",
    "CDRSB",
    "MMSCORE",
]

FEATURE_LABELS = {
    "AGE": "Age",
    "PTGENDER": "Gender",
    "PTEDUCAT": "Education",
    "PTETHCAT": "Ethnicity",
    "PTRACCAT": "Race",
    "APOE": "APOE genotype",
    "CDGLOBAL": "CDR Global",
    "CDRSB": "CDR-SB",
    "MMSCORE": "MMSE",
}

MODEL_FILES = {
    "Random Forest": "random_forest_model.joblib",
    "XGBoost": "xgboost_model.joblib",
    "EBM": "ebm_model.joblib",
}

# ---------------------------------------------------------------------
# Sidebar diagnostics
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 MCI → AD XAI")
    st.caption("Research prototype")

    with st.expander("Runtime diagnostics", expanded=False):
        st.write("Python executable:")
        st.code(sys.executable)

        if interpret is not None:
            st.write("Interpret location:")
            st.code(str(Path(interpret.__file__).resolve()))
        else:
            st.error("Interpret could not be imported.")
            st.exception(INTERPRET_IMPORT_ERROR)

        st.write("SHAP version:")
        st.code(getattr(shap, "__version__", "unknown"))

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def load_joblib(filename):
    """Load a joblib model and show the real exception if loading fails."""
    path = MODEL_DIR / filename

    if not path.exists():
        st.error(f"Missing model file: `{path}`")
        return None

    try:
        return joblib.load(path)
    except Exception as exc:
        st.error(f"Could not load `{filename}`")
        st.exception(exc)
        return None


@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    """Load all trained artifacts once per Streamlit session."""
    models = {}

    # Preprocessor
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"

    if not preprocessor_path.exists():
        st.error(f"Missing preprocessor: `{preprocessor_path}`")
        return None, {}

    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as exc:
        st.error("Could not load `preprocessor.joblib`")
        st.exception(exc)
        return None, {}

    # Explicitly verify Interpret before loading the EBM.
    if INTERPRET_IMPORT_ERROR is not None:
        st.error(
            "The Interpret package could not be imported by this Streamlit process. "
            "The EBM cannot be loaded until this is resolved."
        )
        st.exception(INTERPRET_IMPORT_ERROR)

    for model_name, filename in MODEL_FILES.items():
        model = load_joblib(filename)
        if model is not None:
            models[model_name] = model

    return preprocessor, models


def make_input_dataframe(
    age,
    gender,
    education,
    ethnicity,
    race,
    apoe,
    cdr_global,
    cdrsb,
    mmse,
):
    """Create a one-row dataframe with the exact training feature names."""
    return pd.DataFrame(
        [
            {
                "AGE": age,
                "PTGENDER": gender,
                "PTEDUCAT": education,
                "PTETHCAT": ethnicity,
                "PTRACCAT": race,
                "APOE": apoe,
                "CDGLOBAL": cdr_global,
                "CDRSB": cdrsb,
                "MMSCORE": mmse,
            }
        ],
        columns=FEATURE_ORDER,
    )


def get_processed_feature_names(preprocessor):
    """Get feature names after preprocessing."""
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = []

    cleaned = []
    for name in names:
        # Examples:
        # num__AGE
        # cat__PTGENDER_Female
        cleaned.append(name)

    return cleaned


def original_feature_from_encoded_name(name):
    """
    Map a transformed feature name back to one of the 9 original
    research feature groups.
    """
    text = str(name)

    # Remove ColumnTransformer prefixes.
    if "__" in text:
        text = text.split("__", 1)[1]

    # Exact numerical features.
    if text in FEATURE_ORDER:
        return text

    # One-hot categorical features.
    for feature in CATEGORICAL_FEATURES:
        if text == feature or text.startswith(feature + "_"):
            return feature

    # Fallback: longest matching prefix.
    for feature in FEATURE_ORDER:
        if text.startswith(feature):
            return feature

    return text


def group_shap_values(shap_values, feature_names):
    """
    Convert encoded SHAP values into original feature-group contributions.
    """
    values = np.asarray(shap_values)

    # SHAP versions/model types can produce:
    # (n_features,)
    # (1, n_features)
    # (1, n_features, n_classes)
    # (n_samples, n_features, n_classes)
    if values.ndim == 3:
        # For binary classifiers, use the positive class.
        if values.shape[-1] >= 2:
            values = values[0, :, 1]
        else:
            values = values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]
    elif values.ndim != 1:
        values = values.reshape(-1)

    values = np.asarray(values, dtype=float)

    if len(values) != len(feature_names):
        raise ValueError(
            f"SHAP feature count mismatch: got {len(values)} values "
            f"for {len(feature_names)} processed features."
        )

    grouped = {}

    for value, name in zip(values, feature_names):
        original = original_feature_from_encoded_name(name)
        grouped[original] = grouped.get(original, 0.0) + float(value)

    result = pd.DataFrame(
        [
            {
                "Feature": FEATURE_LABELS.get(feature, feature),
                "Feature_Code": feature,
                "SHAP Contribution": contribution,
                "Absolute Contribution": abs(contribution),
            }
            for feature, contribution in grouped.items()
        ]
    )

    result = result.sort_values(
        "Absolute Contribution",
        ascending=False,
    ).reset_index(drop=True)

    return result


def make_tree_explanation(model, processed, preprocessor):
    """
    Generate a patient-specific SHAP explanation for XGBoost or
    Random Forest.
    """
    try:
        feature_names = get_processed_feature_names(preprocessor)

        if not feature_names:
            feature_names = [f"Feature {i}" for i in range(processed.shape[1])]

        explainer = shap.TreeExplainer(model)
        raw_values = explainer.shap_values(processed)

        # Some SHAP/model combinations return a list for binary classes.
        if isinstance(raw_values, list):
            if len(raw_values) >= 2:
                raw_values = raw_values[1]
            else:
                raw_values = raw_values[0]

        grouped = group_shap_values(raw_values, feature_names)

        return grouped, None

    except Exception as exc:
        return None, exc


def make_ebm_explanation(model, processed, preprocessor):
    """
    Generate a patient-specific EBM explanation.

    The EBM is trained on the same preprocessed 22-variable representation
    used by the primary modeling pipeline. Its native local explanation
    is mapped back to the original nine research feature groups.
    """
    try:
        if not hasattr(model, "explain_local"):
            raise TypeError(
                "The loaded EBM object does not provide explain_local()."
            )

        explanation = model.explain_local(processed)

        # The Interpret explanation API provides data(row_index).
        data = explanation.data(0)

        names = data.get("names", [])
        scores = data.get("scores", [])

        if names is None or scores is None:
            raise ValueError(
                "EBM local explanation did not contain feature names/scores."
            )

        if len(names) != len(scores):
            raise ValueError(
                f"EBM explanation mismatch: {len(names)} names vs "
                f"{len(scores)} scores."
            )

        grouped = {}

        for name, score in zip(names, scores):
            original = original_feature_from_encoded_name(name)
            grouped[original] = grouped.get(original, 0.0) + float(score)

        result = pd.DataFrame(
            [
                {
                    "Feature": FEATURE_LABELS.get(feature, feature),
                    "Feature_Code": feature,
                    "Contribution": contribution,
                    "Absolute Contribution": abs(contribution),
                }
                for feature, contribution in grouped.items()
            ]
        )

        result = result.sort_values(
            "Absolute Contribution",
            ascending=False,
        ).reset_index(drop=True)

        return result, None

    except Exception as exc:
        return None, exc


def plot_contributions(df, value_column, title):
    """Create a horizontal contribution chart."""
    if df is None or df.empty:
        st.info("No explanation values are available.")
        return

    plot_df = df.copy().head(9)
    plot_df = plot_df.sort_values(value_column, ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        plot_df["Feature"],
        plot_df[value_column],
    )

    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Contribution")
    ax.set_ylabel("")
    ax.set_title(title)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def probability_card(probability):
    """Return text category for the research-demo UI."""
    if probability < 0.33:
        return "Lower estimated probability"
    elif probability < 0.66:
        return "Intermediate estimated probability"
    else:
        return "Higher estimated probability"


def model_probability(model, processed):
    """Return positive-class probability."""
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model does not implement predict_proba().")

    probabilities = model.predict_proba(processed)

    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            f"Unexpected predict_proba shape: {probabilities.shape}"
        )

    return float(probabilities[0, 1])


def load_result_csv(filename):
    """Load a result CSV if it exists."""
    path = RESULTS_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------
preprocessor, models = load_models()

# ---------------------------------------------------------------------
# Main navigation
# ---------------------------------------------------------------------
st.title("🧠 MCI → Alzheimer’s Disease Conversion Prediction")

st.markdown(
    """
This research prototype estimates the probability of **AD-related conversion
within 2 years among participants with Mild Cognitive Impairment (MCI)**.

It is **not a diagnostic tool** and does not determine whether a person
currently has Alzheimer’s disease.
"""
)

st.divider()

page = st.sidebar.radio(
    "Navigate",
    [
        "Patient Prediction",
        "Model Performance",
        "Explainability",
        "About",
    ],
)

# ---------------------------------------------------------------------
# Patient Prediction
# ---------------------------------------------------------------------
if page == "Patient Prediction":

    st.header("Patient Prediction")

    st.info(
        "Enter baseline MCI characteristics. The three trained models will "
        "produce estimated probabilities for AD-related conversion within "
        "2 years."
    )

    if preprocessor is None or not models:
        st.error(
            "The trained model artifacts could not be loaded. "
            "Check the Runtime diagnostics in the sidebar."
        )
        st.stop()

    st.subheader("Patient Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=40.0,
            max_value=100.0,
            value=72.0,
            step=1.0,
        )

        education = st.number_input(
            "Education (years)",
            min_value=0.0,
            max_value=30.0,
            value=16.0,
            step=1.0,
        )

        mmse = st.number_input(
            "MMSE",
            min_value=0.0,
            max_value=30.0,
            value=27.0,
            step=1.0,
        )

    with col2:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
        )

        cdr_global = st.number_input(
            "CDR Global",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.5,
        )

        cdrsb = st.number_input(
            "CDR-SB",
            min_value=0.0,
            max_value=18.0,
            value=1.5,
            step=0.5,
        )

    with col3:
        apoe = st.selectbox(
            "APOE genotype",
            [
                "ε3/ε3",
                "ε3/ε4",
                "ε4/ε4",
                "ε2/ε3",
                "ε2/ε4",
            ],
        )

        ethnicity = st.selectbox(
            "Ethnicity",
            [
                "Not Hisp/Latino",
                "Hisp/Latino",
                "Unknown",
            ],
        )

        race = st.selectbox(
            "Race",
            [
                "White",
                "Black",
                "Asian",
                "Other",
                "Unknown",
            ],
        )

    predict_clicked = st.button(
        "🔍 Predict AD Conversion Probability",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        patient_df = make_input_dataframe(
            age=age,
            gender=gender,
            education=education,
            ethnicity=ethnicity,
            race=race,
            apoe=apoe,
            cdr_global=cdr_global,
            cdrsb=cdrsb,
            mmse=mmse,
        )

        try:
            processed = preprocessor.transform(patient_df)
        except Exception as exc:
            st.error("The patient data could not be transformed.")
            st.exception(exc)
            st.stop()

        st.session_state["patient_df"] = patient_df
        st.session_state["processed"] = processed

        st.subheader("Model Predictions")

        prediction_results = []

        for model_name in ["XGBoost", "Random Forest", "EBM"]:
            model = models.get(model_name)

            if model is None:
                prediction_results.append(
                    {
                        "Model": model_name,
                        "Probability": np.nan,
                        "Status": "Model unavailable",
                    }
                )
                continue

            try:
                probability = model_probability(model, processed)

                prediction_results.append(
                    {
                        "Model": model_name,
                        "Probability": probability,
                        "Status": probability_card(probability),
                    }
                )

            except Exception as exc:
                st.error(f"{model_name} prediction failed.")
                st.exception(exc)

                prediction_results.append(
                    {
                        "Model": model_name,
                        "Probability": np.nan,
                        "Status": "Prediction error",
                    }
                )

        results_df = pd.DataFrame(prediction_results)
        st.session_state["prediction_results"] = results_df

        cards = st.columns(3)

        for card, row in zip(cards, prediction_results):
            with card:
                st.metric(
                    row["Model"],
                    (
                        "Unavailable"
                        if pd.isna(row["Probability"])
                        else f"{row['Probability']:.1%}"
                    ),
                )

                st.caption(row["Status"])

        st.divider()

        st.subheader("Prediction Summary")

        display_df = results_df.copy()

        display_df["Estimated probability"] = display_df[
            "Probability"
        ].apply(
            lambda x: (
                "—"
                if pd.isna(x)
                else f"{x:.1%}"
            )
        )

        display_df = display_df[
            ["Model", "Estimated probability", "Status"]
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.warning(
            "Research-use disclaimer: these probabilities are model outputs "
            "from an ADNI-derived research cohort. They are not clinically "
            "validated individual risk estimates and should not be used for "
            "medical diagnosis or treatment decisions."
        )

# ---------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------
elif page == "Model Performance":

    st.header("Model Performance")

    st.markdown(
        """
Performance is reported from the research evaluation. Because the outcome
is imbalanced, ROC-AUC and PR-AUC are particularly relevant alongside
classification metrics.
"""
    )

    performance = load_result_csv("5fold_performance_summary.csv")

    if performance is None:
        performance = load_result_csv("final_model_comparison.csv")

    if performance is not None:
        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "Performance CSV files were not found in the `results/` folder."
        )

    image_candidates = [
        "model_performance_final.png",
    ]

    for image_name in image_candidates:
        image_path = RESULTS_DIR / image_name
        if image_path.exists():
            st.image(
                str(image_path),
                caption="Model performance comparison",
                use_container_width=True,
            )

    st.subheader("Important interpretation note")

    st.markdown(
        """
The EBM showed a high accuracy in the default 0.50-threshold evaluation
because the cohort is imbalanced and the model defaulted to predicting the
majority class. Its ROC-AUC and PR-AUC still provide ranking information.

Therefore, accuracy alone should not be interpreted as evidence of superior
conversion detection.
"""
    )

# ---------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------
elif page == "Explainability":

    st.header("Patient-Specific Explainability")

    st.markdown(
        """
The explanations show which input feature groups contributed most strongly
to each model's prediction for the entered patient.

For XGBoost and Random Forest, explanations are generated with SHAP.
For EBM, the model's native local explanation is used.
"""
    )

    if preprocessor is None or not models:
        st.error(
            "Models are not available. Check the Runtime diagnostics."
        )
        st.stop()

    if "processed" not in st.session_state:
        st.info(
            "First enter patient features on the Patient Prediction page "
            "and click **Predict AD Conversion Probability**."
        )
        st.stop()

    processed = st.session_state["processed"]

    # ---------------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------------
    st.subheader("XGBoost — SHAP Explanation")

    xgb_model = models.get("XGBoost")

    if xgb_model is None:
        st.error("XGBoost model is unavailable.")
    else:
        xgb_df, xgb_error = make_tree_explanation(
            xgb_model,
            processed,
            preprocessor,
        )

        if xgb_error is not None:
            st.error("XGBoost SHAP explanation failed.")
            st.exception(xgb_error)
        else:
            plot_contributions(
                xgb_df,
                "SHAP Contribution",
                "XGBoost patient-specific SHAP contributions",
            )

            st.dataframe(
                xgb_df[
                    [
                        "Feature",
                        "SHAP Contribution",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ---------------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------------
    st.subheader("Random Forest — SHAP Explanation")

    rf_model = models.get("Random Forest")

    if rf_model is None:
        st.error("Random Forest model is unavailable.")
    else:
        rf_df, rf_error = make_tree_explanation(
            rf_model,
            processed,
            preprocessor,
        )

        if rf_error is not None:
            st.error("Random Forest SHAP explanation failed.")
            st.exception(rf_error)
        else:
            plot_contributions(
                rf_df,
                "SHAP Contribution",
                "Random Forest patient-specific SHAP contributions",
            )

            st.dataframe(
                rf_df[
                    [
                        "Feature",
                        "SHAP Contribution",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    # ---------------------------------------------------------------
    # EBM
    # ---------------------------------------------------------------
    st.subheader("Explainable Boosting Machine — Native Explanation")

    ebm_model = models.get("EBM")

    if ebm_model is None:
        st.error(
            "EBM model is unavailable. Check the Runtime diagnostics "
            "in the sidebar."
        )
    else:
        ebm_df, ebm_error = make_ebm_explanation(
            ebm_model,
            processed,
            preprocessor,
        )

        if ebm_error is not None:
            st.error("EBM local explanation failed.")
            st.exception(ebm_error)
        else:
            plot_contributions(
                ebm_df,
                "Contribution",
                "EBM patient-specific feature contributions",
            )

            st.dataframe(
                ebm_df[
                    [
                        "Feature",
                        "Contribution",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    st.subheader("Research-level Cross-Model Consistency")

    consistency = load_result_csv("three_model_pairwise_consistency.csv")
    top_k = load_result_csv("three_model_top_k_consistency.csv")

    if consistency is not None:
        st.markdown("### Pairwise attribution-rank consistency")
        st.dataframe(
            consistency,
            use_container_width=True,
            hide_index=True,
        )

    if top_k is not None:
        st.markdown("### Top-k feature overlap")
        st.dataframe(
            top_k,
            use_container_width=True,
            hide_index=True,
        )

    image_candidates = [
        "three_model_feature_ranking_final.png",
        "cross_model_attribution_final.png",
        "xai_consistency_final.png",
        "xai_stability_final.png",
        "attribution_stability_final.png",
    ]

    for image_name in image_candidates:
        image_path = RESULTS_DIR / image_name
        if image_path.exists():
            st.image(
                str(image_path),
                caption=image_name.replace("_", " ").replace(".png", ""),
                use_container_width=True,
            )

    st.info(
        "The research analysis found strong agreement in feature-attribution "
        "rankings across the three models in this ADNI-derived cohort. "
        "This should not be interpreted as causal evidence or proof of "
        "generalizability to other populations."
    )

# ---------------------------------------------------------------------
# About
# ---------------------------------------------------------------------
elif page == "About":

    st.header("About the Research")

    st.markdown(
        """
### Research title

**Development and Evaluation of Machine Learning Models for Predicting
Conversion from Mild Cognitive Impairment to Alzheimer's Disease:
A Cross-Model Study of Explainable AI Attribution Consistency and Stability**

### Research question

> How consistently and stably do different machine learning models
> attribute features when predicting the conversion of patients with
> Mild Cognitive Impairment to Alzheimer's disease?

### Models

- XGBoost
- Random Forest
- Explainable Boosting Machine (EBM)

### Explainability

- SHAP for XGBoost
- SHAP for Random Forest
- Native local explanations for EBM

### Primary baseline features

- Age
- Gender
- Education
- Ethnicity
- Race
- APOE genotype
- CDR Global
- CDR-SB
- MMSE

### Outcome

The primary prediction target is **AD-related conversion within 2 years**
among participants who had MCI at baseline and met the cohort follow-up
requirements.

### Important limitation

This application is a research demonstration based on an ADNI-derived
dataset and trained research models. It is not a clinical diagnostic system.
It should not be used to diagnose Alzheimer's disease, determine treatment,
or make individual medical decisions.
"""
    )

    st.subheader("Modeling cohort")

    st.markdown(
        """
- Final modeling cohort: **997 participants**
- AD-related converters: **119**
- Non-converters: **878**
- Positive-class prevalence: **11.94%**
- Primary feature groups: **9**
"""
    )

    st.subheader("Interpretation")

    st.markdown(
        """
Feature attribution describes how the trained models used the available
features for prediction. It does **not** establish that a feature causes
Alzheimer's disease or causes conversion from MCI to Alzheimer's disease.

The cross-model analysis evaluates consistency of model explanations within
this research cohort; it does not establish universal clinical validity.
"""
    )

    st.caption(
        "Research prototype • MCI → AD conversion within 2 years • "
        "Not for clinical diagnosis"
    )
