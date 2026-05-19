import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from PIL import Image

from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="Clinical Trial Outcome Predictor",
    page_icon="🧬",
    layout="wide"
)

st.markdown(
    """
    <style>
    a[href^="#"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #3D7391;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    .section-box {
        padding: 18px 18px 10px 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(128, 128, 128, 0.06);
        margin-bottom: 20px;
    }
    
    .section-box {
        padding: 18px 18px 10px 18px;
        border-radius: 14px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        background-color: rgba(128, 128, 128, 0.06);
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 14px;
    }

    .result-heading {
        padding: 12px 16px;
        border-radius: 12px;
        border-left: 5px solid #4F8BF9;
        background-color: rgba(128, 128, 128, 0.08);
        font-size: 20px;
        font-weight: 700;
        margin-top: 24px;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# FeatureBuilder class
# This must be defined before loading .pkl files
# --------------------------------------------------

class FeatureBuilder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Store sponsor frequency from the training data.
        sponsor = X["Sponsor"].fillna("Unknown")
        self.sponsor_counts_ = sponsor.value_counts()
        return self

    def transform(self, X):
        # Create the same engineered features used during training.
        X = X.copy()

        for col in ["Sex", "Funder Type", "Study Design"]:
            X[col] = X[col].fillna("Unknown")

        X["Conditions"] = X["Conditions"].fillna("")
        X["Interventions"] = X["Interventions"].fillna("")

        # Count condition terms separated by |.
        X["condition_count"] = X["Conditions"].apply(
            lambda x: len([v for v in str(x).split("|") if v.strip()])
        )

        # Count intervention terms separated by |.
        X["intervention_count"] = X["Interventions"].apply(
            lambda x: len([v for v in str(x).split("|") if v.strip()])
        )

        loc = X["Locations"]
        loc_text = loc.fillna("").str.lower()

        X["has_location"] = loc.notna().astype(int)

        X["location_count"] = loc.fillna("").apply(
            lambda x: len([v for v in str(x).split("|") if v.strip()])
        )

        X["is_multisite"] = (X["location_count"] > 1).astype(int)

        X["has_us_location"] = loc_text.str.contains("united states", regex=False).astype(int)
        X["has_uk_location"] = loc_text.str.contains("united kingdom", regex=False).astype(int)
        X["has_china_location"] = loc_text.str.contains("china", regex=False).astype(int)
        X["has_canada_location"] = loc_text.str.contains("canada", regex=False).astype(int)

        def has_age_group(value, group):
            values = [v.strip() for v in str(value).upper().split(",")]
            return int(group in values)

        X["has_child"] = X["Age"].apply(lambda x: has_age_group(x, "CHILD"))
        X["has_adult"] = X["Age"].apply(lambda x: has_age_group(x, "ADULT"))
        X["has_older_adult"] = X["Age"].apply(lambda x: has_age_group(x, "OLDER_ADULT"))

        def has_phase(value, phase):
            values = [v.strip() for v in str(value).upper().split("|")]
            return int(phase in values)

        X["has_early_phase1"] = X["Phases"].apply(lambda x: has_phase(x, "EARLY_PHASE1"))
        X["has_phase1"] = X["Phases"].apply(lambda x: has_phase(x, "PHASE1"))
        X["has_phase2"] = X["Phases"].apply(lambda x: has_phase(x, "PHASE2"))
        X["has_phase3"] = X["Phases"].apply(lambda x: has_phase(x, "PHASE3"))
        X["has_phase4"] = X["Phases"].apply(lambda x: has_phase(x, "PHASE4"))

        intervention = X["Interventions"].fillna("").str.upper()

        X["has_drug"] = intervention.str.contains("DRUG:", regex=False).astype(int)
        X["has_device"] = intervention.str.contains("DEVICE:", regex=False).astype(int)
        X["has_biological"] = intervention.str.contains("BIOLOGICAL:", regex=False).astype(int)
        X["has_procedure"] = intervention.str.contains("PROCEDURE:", regex=False).astype(int)
        X["has_behavioral"] = intervention.str.contains("BEHAVIORAL:", regex=False).astype(int)

        X["has_diagnostic_test"] = (
            intervention.str.contains("DIAGNOSTIC_TEST:", regex=False) |
            intervention.str.contains("DIAGNOSTIC TEST:", regex=False)
        ).astype(int)

        # Sponsor is simplified in the deployed app.
        sponsor = X["Sponsor"].fillna("Unknown")
        X["sponsor_frequency"] = sponsor.map(self.sponsor_counts_).fillna(1)

        def get_design_value(text, key):
            for part in str(text).split("|"):
                part = part.strip()
                if part.upper().startswith(key.upper() + ":"):
                    return part.split(":", 1)[1].strip()
            return "Unknown"

        # Extract study design parts.
        X["allocation"] = X["Study Design"].apply(lambda x: get_design_value(x, "Allocation"))
        X["intervention_model"] = X["Study Design"].apply(lambda x: get_design_value(x, "Intervention Model"))
        X["masking"] = X["Study Design"].apply(lambda x: get_design_value(x, "Masking"))
        X["primary_purpose"] = X["Study Design"].apply(lambda x: get_design_value(x, "Primary Purpose"))

        return X


# --------------------------------------------------
# Load saved model files
# --------------------------------------------------

@st.cache_resource
def load_files():
    completion_builder = joblib.load("feature_builder.pkl")
    completion_raw_cols = joblib.load("raw_cols_to_drop.pkl")
    completion_preprocessor = joblib.load("completion_preprocessor.pkl")
    completion_model = joblib.load("completion_xgboost_model.pkl")

    duration_builder = joblib.load("duration_feature_builder.pkl")
    duration_raw_cols = joblib.load("duration_raw_cols_to_drop.pkl")
    duration_preprocessor = joblib.load("duration_preprocessor.pkl")
    duration_model = joblib.load("duration_xgboost_model.pkl")

    return (
        completion_builder,
        completion_raw_cols,
        completion_preprocessor,
        completion_model,
        duration_builder,
        duration_raw_cols,
        duration_preprocessor,
        duration_model
    )


(
    completion_builder,
    completion_raw_cols,
    completion_preprocessor,
    completion_model,
    duration_builder,
    duration_raw_cols,
    duration_preprocessor,
    duration_model
) = load_files()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

COMPLETION_THRESHOLD = 0.50


def build_locations(location_count, location_countries):
    # Build location text in the same | style used in the dataset.
    if location_count == 0 or len(location_countries) == 0:
        return ""

    locations = []

    for i in range(int(location_count)):
        locations.append(location_countries[i % len(location_countries)])

    return " | ".join(locations)


def build_interventions(intervention_types, intervention_keywords):
    # Build intervention text so type flags and text features can both work.
    terms = [v.strip() for v in str(intervention_keywords).split("|") if v.strip()]

    if len(intervention_types) == 0:
        return " | ".join(terms)

    if len(terms) == 0:
        return " | ".join([f"{t}: Unknown" for t in intervention_types])

    interventions = []

    for intervention_type in intervention_types:
        for term in terms:
            interventions.append(f"{intervention_type}: {term}")

    return " | ".join(interventions)


def build_raw_input(
    enrollment,
    start_year,
    sex,
    phases,
    funder_type,
    location_count,
    location_countries,
    conditions,
    intervention_types,
    intervention_keywords,
    age_groups,
    allocation,
    intervention_model,
    masking,
    primary_purpose
):
    # Convert app inputs into the raw column format expected by the saved pipeline.
    locations = build_locations(location_count, location_countries)

    age_text = ", ".join(age_groups)

    phase_text = " | ".join(phases) if len(phases) > 0 else "NA"

    interventions = build_interventions(
        intervention_types=intervention_types,
        intervention_keywords=intervention_keywords
    )

    # Sponsor is kept simple to avoid unreliable free-text sponsor entries.
    sponsor = "Unknown"

    study_design = (
        f"Allocation: {allocation} | "
        f"Intervention Model: {intervention_model} | "
        f"Masking: {masking} | "
        f"Primary Purpose: {primary_purpose}"
    )

    input_df = pd.DataFrame({
        "Enrollment": [enrollment],
        "start_year": [start_year],
        "Sex": [sex],
        "Funder Type": [funder_type],
        "Sponsor": [sponsor],
        "Locations": [locations],
        "Conditions": [conditions],
        "Interventions": [interventions],
        "Age": [age_text],
        "Phases": [phase_text],
        "Study Design": [study_design]
    })

    return input_df


def clean_feature_name(feature):
    # Make technical feature names easier to understand in the app.
    feature = str(feature)

    if feature.startswith("numeric__"):
        name = feature.replace("numeric__", "")

    elif feature.startswith("binary__"):
        name = feature.replace("binary__", "")

    elif feature.startswith("categorical__"):
        name = feature.replace("categorical__", "")

    elif feature.startswith("conditions_text__"):
        term = feature.replace("conditions_text__", "").replace("_", " ")
        return "Condition text signal: " + term

    elif feature.startswith("interventions_text__"):
        term = feature.replace("interventions_text__", "").replace("_", " ")
        return "Intervention text signal: " + term

    else:
        name = feature

    name = name.replace("_", " ")

    readable_names = {
        "Enrollment": "Enrollment",
        "start year": "Start year",
        "condition count": "Condition count",
        "intervention count": "Intervention count",
        "location count": "Location count",
        "sponsor frequency": "Sponsor frequency",

        "has location": "Has location",
        "is multisite": "Multisite trial",
        "has us location": "US location",
        "has uk location": "UK location",
        "has china location": "China location",
        "has canada location": "Canada location",

        "has child": "Includes children",
        "has adult": "Includes adults",
        "has older adult": "Includes older adults",

        "has early phase1": "Early Phase 1",
        "has phase1": "Phase 1",
        "has phase2": "Phase 2",
        "has phase3": "Phase 3",
        "has phase4": "Phase 4",

        "has drug": "Drug intervention",
        "has device": "Device intervention",
        "has biological": "Biological intervention",
        "has procedure": "Procedure intervention",
        "has behavioral": "Behavioural intervention",
        "has diagnostic test": "Diagnostic test intervention",

        "Sex MALE": "Sex: Male",
        "Sex FEMALE": "Sex: Female",
        "Sex ALL": "Sex: All",

        "Funder Type INDUSTRY": "Funder type: Industry",
        "Funder Type OTHER": "Funder type: Other",
        "Funder Type NIH": "Funder type: NIH",
        "Funder Type FED": "Funder type: Federal",
        "Funder Type NETWORK": "Funder type: Network",
        "Funder Type INDIV": "Funder type: Individual",
    }

    name = readable_names.get(name, name)

    name = name.replace("allocation ", "Allocation: ")
    name = name.replace("intervention model ", "Intervention model: ")
    name = name.replace("masking ", "Masking: ")
    name = name.replace("primary purpose ", "Primary purpose: ")

    return name.strip()


def format_feature_value(feature, engineered_row):
    # Add values beside numeric SHAP features where possible.
    feature = str(feature)

    if feature.startswith("numeric__"):
        raw_name = feature.replace("numeric__", "")

        if raw_name in engineered_row.index:
            value = engineered_row[raw_name]
        else:
            return ""

        if raw_name == "Enrollment":
            return f": {int(round(value))} participants"
        elif raw_name == "location_count":
            return f": {int(round(value))} locations"
        elif raw_name == "condition_count":
            return f": {int(round(value))} condition terms"
        elif raw_name == "intervention_count":
            return f": {int(round(value))} intervention terms"
        elif raw_name == "start_year":
            return f": {int(round(value))}"
        else:
            return f": {round(float(value), 2)}"

    return ""


def get_shap_explanation(X_completion, completion_features, top_n=5):
    # Create a short explanation for the individual prediction.
    if hasattr(X_completion, "toarray"):
        X_dense = X_completion.toarray()
    else:
        X_dense = X_completion

    feature_names = completion_preprocessor.get_feature_names_out()
    X_shap_df = pd.DataFrame(X_dense, columns=feature_names)

    explainer = shap.TreeExplainer(completion_model)
    shap_values = explainer.shap_values(X_shap_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap_values_single = shap_values[0]

    shap_table = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": X_shap_df.iloc[0].values,
        "SHAP Value": shap_values_single,
        "Contribution Strength": np.abs(shap_values_single)
    })

    # Hide absent condition/intervention text signals.
    # This keeps the SHAP explanation easier to understand for users.
    text_signal = (
            shap_table["Feature"].str.startswith("conditions_text__") |
            shap_table["Feature"].str.startswith("interventions_text__")
    )

    shap_table = shap_table[
        ~(text_signal & (shap_table["Feature Value"] == 0))
    ]

    shap_table = shap_table.sort_values(
        by="Contribution Strength",
        ascending=False
    )
    engineered_row = completion_features.iloc[0]

    explanation_lines = []

    for _, row in shap_table.head(top_n).iterrows():
        readable_feature = clean_feature_name(row["Feature"])
        value_text = format_feature_value(row["Feature"], engineered_row)

        if row["SHAP Value"] > 0:
            direction = "pushed toward Completed"
        else:
            direction = "pushed toward Not Completed"

        explanation_lines.append(
            f"{readable_feature}{value_text} — {direction}"
        )

    return explanation_lines


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🧪 Clinical Trial Dashboard")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Predict Trial Outcome", "About the Model"]
)
st.sidebar.image(
    "side_bar.jpg",
    use_container_width=True
)

# --------------------------------------------------
# Home page
# --------------------------------------------------

if page == "Home":

    st.markdown(
        """
        <style>
        .title-box {
            padding: 24px;
            border-radius: 16px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background-color: rgba(128, 128, 128, 0.08);
            margin-top: 18px;
            margin-bottom: 28px;
        }

        .title-box h1 {
            margin-bottom: 8px;
        }

        .title-box p {
            font-size: 17px;
            color: gray;
            line-height: 1.5;
            margin-bottom: 0;
        }

        .info-card {
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            background-color: rgba(128, 128, 128, 0.08);
            height: 210px;
            margin-bottom: 15px;
        }

        .info-card h3 {
            font-size: 19px;
            margin-top: 0;
            margin-bottom: 10px;
        }

        .info-card p {
            font-size: 15px;
            line-height: 1.5;
            margin-bottom: 0;
        }

        .small-disclaimer {
            font-size: 14px;
            color: gray;
            line-height: 1.5;
            margin-top: 35px;
            padding-top: 12px;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    try:
        image = Image.open("clinical_trials.png")
        image = image.resize((1200, 400))
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.info("Add clinical_trials.png to the app folder to show the header image.")

    st.markdown(
        """
        <div class="title-box">
            <h1>Clinical Trial Outcome Predictor</h1>
            <p>
                A machine learning web app for exploring clinical trial completion likelihood 
                and expected trial duration.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🏥 About Clinical Trials")

    st.write(
        "Clinical trials are research studies that test whether medical treatments, drugs, "
        "devices, or procedures are safe and effective before they are used more widely in healthcare. "
        "They are important because they provide evidence for treatment development, medical "
        "decision-making, and regulatory approval."
    )

    st.write(
        "However, clinical trials can be difficult to complete. They may fail for several reasons, "
        "including lack of treatment efficacy, safety issues, funding problems, and challenges with "
        "patient recruitment and retention."
    )

    st.markdown("---")

    st.markdown("### 📌 About This App")

    st.write(
        "This web app was developed using over 60,000 interventional clinical trial records "
        "from ClinicalTrials.gov, covering trials that started between January 2015 and December 2025."

    )

    st.write(
        "The app helps users explore possible clinical trial outcomes by estimating whether a trial is likely "
        "to be completed and how long it may take. It also highlights the main factors behind the result in a "
        "clear and understandable way."

    )

    st.markdown("---")

    st.markdown("### ⚙️ What the App Does")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>1️⃣ Completion Prediction</h3>
                <p>Predicts whether a clinical trial is likely to be completed or not completed.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>2️⃣ Duration Estimation</h3>
                <p>Estimates the expected clinical trial duration only when the trial is predicted as completed.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h3>3️⃣ SHAP Explainability</h3>
                <p>Shows the main features influencing the completion prediction.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="small-disclaimer">
            <strong>⚠️ Disclaimer:</strong> This dashboard is an educational machine learning prototype. 
            Predictions should be interpreted as model-based risk signals, not as clinical, regulatory, 
            or operational decisions.
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# Prediction page
# --------------------------------------------------

elif page == "Predict Trial Outcome":

    st.title("Predict Clinical Trial Outcome")
    st.write("Enter trial details below to generate a completion prediction, duration estimate, and SHAP explanation.")

    with st.expander("Enter Trial Details", expanded=True):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Trial Size and Timing</div>', unsafe_allow_html=True)

            enrollment = st.number_input(
                "Enrollment",
                min_value=1,
                value=100,
                help="Number of participants planned or enrolled in the trial."
            )

            start_year = st.number_input(
                "Planned / Actual Start Year",
                min_value=2015,
                max_value=2025,
                value=2024,
                help="Start year is used as historical context within the 2015–2025 training data, not to forecast future years."
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Eligibility</div>', unsafe_allow_html=True)

            sex = st.selectbox(
                "Sex Eligibility",
                ["ALL", "MALE", "FEMALE"]
            )

            age_groups = st.multiselect(
                "Age Groups",
                ["CHILD", "ADULT", "OLDER_ADULT"],
                default=["ADULT"]
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Study Design</div>', unsafe_allow_html=True)

            phases = st.multiselect(
                "Trial Phase",
                ["NA", "EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4"],
                default=["NA"],
                help="Select all phases that apply."
            )

            allocation = st.selectbox(
                "Allocation",
                ["Unknown", "Randomized", "Non-Randomized", "N/A"]
            )

            intervention_model = st.selectbox(
                "Intervention Model",
                [
                    "Unknown",
                    "Parallel Assignment",
                    "Single Group Assignment",
                    "Crossover Assignment",
                    "Sequential Assignment",
                    "Factorial Assignment"
                ]
            )

            masking = st.selectbox(
                "Masking",
                ["Unknown", "None (Open Label)", "Single", "Double", "Triple", "Quadruple"]
            )

            primary_purpose = st.selectbox(
                "Primary Purpose",
                [
                    "Unknown",
                    "Treatment",
                    "Prevention",
                    "Diagnostic",
                    "Supportive Care",
                    "Screening",
                    "Health Services Research",
                    "Basic Science",
                    "Other"
                ]
            )

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Funding and Location</div>', unsafe_allow_html=True)

            funder_type = st.selectbox(
                "Funder Type",
                ["OTHER", "INDUSTRY", "NIH", "FED", "NETWORK", "INDIV"]
            )

            location_count = st.number_input(
                "Number of Locations",
                min_value=0,
                value=1,
                help="Total number of trial locations."
            )

            location_countries = st.multiselect(
                "Trial Location Countries",
                ["United States", "United Kingdom", "China", "Canada", "Other"],
                default=["Other"],
                help="Select all main countries involved in the trial."
            )

            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-box">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Clinical Information</div>', unsafe_allow_html=True)

            conditions = st.text_area(
                "Condition Keywords",
                value="",
                placeholder="Example: Hepatitis C|HIV Infection",
                help="Enter short condition keywords separated by |. Avoid long paragraphs."
            )

            intervention_types = st.multiselect(
                "Intervention Types",
                [
                    "DRUG",
                    "DEVICE",
                    "BIOLOGICAL",
                    "PROCEDURE",
                    "BEHAVIORAL",
                    "DIAGNOSTIC_TEST",
                    "OTHER"
                ],
                default=[],
                help="Select all intervention types that apply."
            )

            intervention_keywords = st.text_area(
                "Intervention Keywords",
                value="",
                placeholder="Example: DRUG: Indocyanine Green|DEVICE: MultiSpectral Imaging System",
                help="Enter short intervention keywords separated by |. Avoid long paragraphs."
            )

            st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict"):

        raw_input = build_raw_input(
            enrollment=enrollment,
            start_year=start_year,
            sex=sex,
            phases=phases,
            funder_type=funder_type,
            location_count=location_count,
            location_countries=location_countries,
            conditions=conditions,
            intervention_types=intervention_types,
            intervention_keywords=intervention_keywords,
            age_groups=age_groups,
            allocation=allocation,
            intervention_model=intervention_model,
            masking=masking,
            primary_purpose=primary_purpose
        )

        # Apply the saved feature engineering and preprocessing.
        completion_features = completion_builder.transform(raw_input)
        completion_features = completion_features.drop(columns=completion_raw_cols, errors="ignore")
        X_completion = completion_preprocessor.transform(completion_features)

        # Predict completion probability.
        completion_probs = completion_model.predict_proba(X_completion)[0]
        completion_probability = completion_probs[1]
        non_completion_probability = completion_probs[0]

        if completion_probability >= COMPLETION_THRESHOLD:
            predicted_outcome = "Completed"
            probability_label = "Completion probability"
            probability_value = completion_probability
        else:
            predicted_outcome = "Not Completed"
            probability_label = "Non-completion probability"
            probability_value = non_completion_probability

        st.markdown("---")
        st.markdown('<div class="result-heading">Prediction Outcome</div>', unsafe_allow_html=True)

        if predicted_outcome == "Completed":
            # Duration is shown only for trials predicted as completed.
            duration_features = duration_builder.transform(raw_input)
            duration_features = duration_features.drop(columns=duration_raw_cols, errors="ignore")
            X_duration = duration_preprocessor.transform(duration_features)

            estimated_duration = duration_model.predict(X_duration)[0]
            estimated_duration = max(0, estimated_duration)

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted outcome", predicted_outcome)

            with result_col2:
                st.metric(probability_label, f"{probability_value * 100:.1f}%")

            st.markdown('<div class="result-heading">Estimated Duration</div>', unsafe_allow_html=True)
            st.metric("Estimated duration", f"{estimated_duration:.1f} months")

        else:
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted outcome", predicted_outcome)

            with result_col2:
                st.metric(probability_label, f"{probability_value * 100:.1f}%")

            st.markdown('<div class="result-heading">Estimated Duration</div>', unsafe_allow_html=True)
            st.info("Duration is not estimated because this trial is predicted as not completed.")

        st.markdown('<div class="result-heading">SHAP Explainability</div>', unsafe_allow_html=True)

        explanation_lines = get_shap_explanation(
            X_completion=X_completion,
            completion_features=completion_features,
            top_n=5
        )

        for i, line in enumerate(explanation_lines, start=1):
            st.write(f"{i}. {line}")

        st.caption(
            "SHAP explains how the model used the input features for this prediction. "
            "It does not prove the real-world reason for trial completion or non-completion."
        )


# --------------------------------------------------
# About model page
# --------------------------------------------------

elif page == "About the Model":

    st.title("About the Model")

    st.caption(
        "Technical summary of the models used for clinical trial outcome prediction and duration estimation."
    )

    st.markdown("---")

    st.markdown("### 1️⃣ Models Used")

    st.markdown(
        """
        This dashboard uses two **XGBoost** models trained on clinical trial records from 
        **ClinicalTrials.gov**, covering trials with start dates from **2015 to 2025**.

        XGBoost was selected as the final modelling approach after comparison with simpler baseline models. 
        Logistic Regression was tested for completion prediction, while Linear Regression was tested for 
        duration estimation.

        XGBoost was used because it can capture non-linear patterns and interactions between trial 
        characteristics such as enrollment size, phase, funding type, location, eligibility, study design, 
        conditions, and interventions.
        """
    )

    st.markdown("---")

    st.markdown("### 2️⃣ Completion Prediction")

    st.markdown(
        """
        The completion model predicts whether a clinical trial is likely to be **completed** or 
        **not completed**.

        The result is displayed as completion or non-completion probability, making the output easier 
        to interpret.
        """
    )

    st.markdown("---")

    st.markdown("### 3️⃣ Duration Estimation")

    st.markdown(
        """
        The duration model estimates the expected clinical trial duration in months.

        This estimate is shown only when the trial is predicted as completed, because the duration model 
        was trained using completed trials with valid recorded durations.
        """
    )

    st.markdown("---")

    st.markdown("### 4️⃣ Model Inputs")

    st.markdown(
        """
        The deployed app uses key trial-level information including enrollment, start year, sex eligibility, 
        age groups, phase, funding type, trial locations, conditions, interventions, and selected study 
        design details.

        Conditions and interventions are entered as short keywords separated by **|**, which follows the 
        format used during feature engineering. Sponsor is simplified as **Unknown** in the prototype to 
        avoid unreliable free-text sponsor entries.
        """
    )

    st.markdown("---")

    st.markdown("### 5️⃣ Explainability")

    st.markdown(
        """
        SHAP (**SHapley Additive exPlanations**) breaks down each prediction by showing how much each feature 
        contributed to the final result, either increasing or decreasing the predicted completion probability.

        This makes the model output easier to interpret by showing the main factors behind each prediction.
        """
    )

    st.markdown(
        """
        <div style="
            font-size: 14px;
            color: gray;
            line-height: 1.5;
            margin-top: 35px;
            padding-top: 12px;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
        ">
            <strong>Disclaimer:</strong> This dashboard is an educational machine learning prototype. 
            Predictions are model-based signals and should not be used for clinical, regulatory, 
            or operational decisions.
        </div>
        """,
        unsafe_allow_html=True
    )
