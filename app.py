# from pathlib import Path
# import json

# import joblib
# import numpy as np
# import pandas as pd
# import streamlit as st


# ROOT = Path(__file__).resolve().parent
# MODEL_DIR = ROOT / "models_v2"

# MODEL_NAMES = {
#     "naive_bayes": "Multinomial Naive Bayes",
#     "logistic_regression": "Logistic Regression",
#     "linear_svm": "Linear SVM",
#     "random_forest": "Random Forest",
#     "xgboost": "XGBoost",
# }

# st.set_page_config(
#     page_title="Multi-Model Spam Detection",
#     page_icon="📧",
#     layout="wide",
# )


# @st.cache_resource
# def load_artifacts():
#     with (MODEL_DIR / "configurations.json").open(
#         encoding="utf-8"
#     ) as file:
#         configurations = json.load(file)

#     vectorizer = joblib.load(MODEL_DIR / "tfidf.joblib")

#     names = {
#         name
#         for configuration in configurations
#         for name in configuration["models"]
#     }

#     models = {
#         name: joblib.load(MODEL_DIR / f"{name}.joblib")
#         for name in names
#     }

#     return vectorizer, models, configurations


# def configuration_name(configuration):
#     names = [
#         MODEL_NAMES.get(name, name)
#         for name in configuration["models"]
#     ]
#     return " + ".join(names)


# def individual_probabilities(features, models):
#     probabilities = {}

#     for name, model in models.items():
#         spam_index = list(model.classes_).index(1)
#         probabilities[name] = float(
#             model.predict_proba(features)[0, spam_index]
#         )

#     return probabilities


# def configuration_probability(configuration, probabilities):
#     return float(np.mean([
#         probabilities[name]
#         for name in configuration["models"]
#     ]))


# st.title("📧 Multi-Model Spam Detection")
# st.write(
#     "Analyze an email using five individual models "
#     "or any of their 26 hybrid combinations."
# )

# try:
#     vectorizer, models, configurations = load_artifacts()
# except Exception as exc:
#     st.error(
#         "Could not load the trained models. "
#         "Complete the training step first."
#     )
#     st.code(str(exc))
#     st.stop()

# configuration_lookup = {
#     configuration["id"]: configuration
#     for configuration in configurations
# }

# with st.sidebar:
#     st.header("Model selection")

#     selected_id = st.selectbox(
#         "Select a model or combination",
#         options=list(configuration_lookup),
#         index=list(configuration_lookup).index("linear_svm"),
#         format_func=lambda key: configuration_name(
#             configuration_lookup[key]
#         ),
#     )

#     selected = configuration_lookup[selected_id]

#     st.caption(
#         f"{len(selected['models'])} model(s) selected. "
#         "Hybrids use equal-weight probability averaging."
#     )
#     st.write(f"Available configurations: **{len(configurations)}**")
#     st.caption(
#         "Scores are model estimates, not guarantees. "
#         "The training corpus contains older English emails."
#     )

# with st.form("email_form"):
#     subject = st.text_input("Email subject")
#     body = st.text_area(
#         "Email body",
#         height=250,
#         placeholder="Paste the email text here...",
#     )

#     compare_all = st.checkbox("Compare all 31 configurations")
#     submitted = st.form_submit_button(
#         "Analyze email",
#         type="primary",
#     )

# if submitted:
#     # Matches the training script's subject/body concatenation.
#     text = " ".join(f"{subject} {body}".split())

#     if not text:
#         st.warning("Enter an email subject or body.")
#         st.stop()

#     features = vectorizer.transform([text])

#     if features.nnz == 0:
#         st.warning(
#             "This email contains no terms recognized by the "
#             "trained vocabulary. A useful prediction is unavailable."
#         )
#         st.stop()

#     # Run only the models needed for the requested operation.
#     required_names = (
#         set(models)
#         if compare_all
#         else set(selected["models"])
#     )

#     with st.spinner("Analyzing email..."):
#         probabilities = individual_probabilities(
#             features,
#             {
#                 name: models[name]
#                 for name in required_names
#             },
#         )

#     spam_probability = configuration_probability(
#         selected, probabilities
#     )
#     threshold = selected.get("threshold", 0.5)
#     is_spam = spam_probability >= threshold

#     st.subheader("Prediction")

#     if is_spam:
#         st.error("SPAM")
#     else:
#         st.success("HAM — legitimate email")

#     left, right = st.columns(2)
#     left.metric("Spam score", f"{spam_probability:.1%}")
#     right.metric("Ham score", f"{1 - spam_probability:.1%}")

#     st.caption(
#         f"Selected: {configuration_name(selected)}. "
#         f"Spam threshold: {threshold:.0%}."
#     )

#     st.subheader("Individual model scores")
#     member_rows = [
#         {
#             "Model": MODEL_NAMES[name],
#             "Spam score": probabilities[name],
#         }
#         for name in selected["models"]
#     ]

#     st.dataframe(
#         pd.DataFrame(member_rows),
#         column_config={
#             "Spam score": st.column_config.NumberColumn(
#                 format="%.4f"
#             )
#         },
#         hide_index=True,
#         use_container_width=True,
#     )

#     if compare_all:
#         rows = []

#         for configuration in configurations:
#             score = configuration_probability(
#                 configuration, probabilities
#             )
#             rows.append({
#                 "Configuration": configuration_name(configuration),
#                 "Type": configuration["type"],
#                 "Prediction": (
#                     "SPAM"
#                     if score >= configuration.get("threshold", 0.5)
#                     else "HAM"
#                 ),
#                 "Spam score": score,
#             })

#         comparison = pd.DataFrame(rows)

#         st.subheader("All 31 predictions")
#         st.dataframe(
#             comparison,
#             hide_index=True,
#             use_container_width=True,
#         )

#         st.download_button(
#             "Download prediction comparison",
#             data=comparison.to_csv(index=False).encode("utf-8"),
#             file_name="email_predictions.csv",
#             mime="text/csv",
#         )






from pathlib import Path
import importlib.util
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models_v2"
MAX_BATCH = 500

DISPLAY_NAMES = {
    "naive_bayes": "Multinomial Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

st.set_page_config(
    page_title="Multi-Model Spam Detection",
    page_icon="📧",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    with (MODEL_DIR / "configurations.json").open(
        encoding="utf-8"
    ) as file:
        configurations = json.load(file)

    vectorizer = joblib.load(MODEL_DIR / "tfidf.joblib")

    names = {
        name
        for configuration in configurations
        for name in configuration["models"]
    }

    models = {
        name: joblib.load(MODEL_DIR / f"{name}.joblib")
        for name in names
    }

    # Load the parser snapshot saved during V2 training.
    parser_path = MODEL_DIR / "prepare_dataset_snapshot.py"
    spec = importlib.util.spec_from_file_location(
        "v2_email_parser", parser_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the V2 parser snapshot.")

    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)

    return vectorizer, models, configurations, parser


def display_name(configuration):
    return " + ".join(
        DISPLAY_NAMES.get(name, name)
        for name in configuration["models"]
    )


def analyze(records, configurations, vectorizer, models):
    """Return one result row per email and configuration."""
    texts = records["text"].fillna("").astype(str).tolist()
    features = vectorizer.transform(texts)

    empty = np.array([not text.strip() for text in texts])
    unknown = np.asarray(features.getnnz(axis=1)).ravel() == 0
    usable = ~(empty | unknown)

    required_names = {
        name
        for configuration in configurations
        for name in configuration["models"]
    }

    probabilities = {}

    if usable.any():
        for name in required_names:
            model = models[name]
            spam_index = list(model.classes_).index(1)

            scores = np.full(len(texts), np.nan)
            scores[usable] = model.predict_proba(
                features[usable]
            )[:, spam_index]

            probabilities[name] = scores

    frames = []

    for configuration in configurations:
        scores = np.full(len(texts), np.nan)

        if usable.any():
            scores = np.mean(
                [
                    probabilities[name]
                    for name in configuration["models"]
                ],
                axis=0,
            )

        threshold = configuration.get("threshold", 0.5)

        predictions = np.full(
            len(texts), "UNAVAILABLE", dtype=object
        )
        predictions[usable] = np.where(
            scores[usable] >= threshold, "SPAM", "HAM"
        )

        status = np.full(len(texts), "OK", dtype=object)
        status[unknown] = "No recognized vocabulary"
        status[empty] = "Empty email"

        frame = pd.DataFrame({
            "email_number": np.arange(1, len(texts) + 1),
            "subject": (
                records["subject"].fillna("").astype(str).to_numpy()
            ),
            "configuration": display_name(configuration),
            "prediction": predictions,
            "spam_score": scores,
            "threshold": threshold,
            "status": status,
        })
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def save_result(key, records, configurations):
    with st.spinner("Analyzing emails..."):
        result = analyze(
            records, configurations, vectorizer, models
        )

    st.session_state[key] = result


def show_result(key):
    result = st.session_state.get(key)
    if result is None:
        return

    st.subheader("Results")
    st.caption(
        "These results reflect the configuration selected when "
        "Analyze was clicked. Scores are estimates, not guarantees."
    )

    st.dataframe(
        result,
        column_config={
            "spam_score": st.column_config.NumberColumn(
                "Spam score", format="%.4f"
            ),
            "threshold": st.column_config.NumberColumn(
                "Threshold", format="%.2f"
            ),
        },
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download results CSV",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


st.title("📧 Multi-Model Spam Detection")
st.write(
    "Analyze pasted emails, upload an email file, "
    "or classify a batch of emails."
)

try:
    vectorizer, models, configurations, parser = load_artifacts()
except Exception as exc:
    st.error("Could not load the V2 models or parser.")
    st.code(str(exc))
    st.stop()

lookup = {
    configuration["id"]: configuration
    for configuration in configurations
}
configuration_ids = list(lookup)

with st.sidebar:
    st.header("Model selection")

    default_index = (
        configuration_ids.index("linear_svm")
        if "linear_svm" in lookup
        else 0
    )

    selected_id = st.selectbox(
        "Model or combination",
        options=configuration_ids,
        index=default_index,
        format_func=lambda key: display_name(lookup[key]),
    )

    compare_all = st.checkbox(
        "Compare all 31 configurations",
        value=False,
    )

    st.caption(
        "Hybrids average their member models' spam scores."
    )
    st.write("**Model version:** V2")
    st.write(f"**Configurations:** {len(configurations)}")
    st.caption(
        "Linear SVM scored 98.87% accuracy on the project's "
        "888-email test split. Performance on new email may differ."
    )

active_configurations = (
    configurations if compare_all else [lookup[selected_id]]
)

manual_tab, eml_tab, batch_tab = st.tabs([
    "Paste email",
    "Upload .eml",
    "Batch CSV",
])

with manual_tab:
    with st.form("manual_form"):
        subject = st.text_input("Subject")
        body = st.text_area("Email body", height=220)
        analyze_manual = st.form_submit_button(
            "Analyze email", type="primary"
        )

    if analyze_manual:
        # Apply the training parser's text normalization.
        normalized_subject = parser.clean_text(subject)
        normalized_body = parser.clean_text(body)

        records = pd.DataFrame([{
            "subject": normalized_subject,
            "text": parser.clean_text(
                f"{normalized_subject} {normalized_body}"
            ),
        }])

        save_result(
            "manual_predictions",
            records,
            active_configurations,
        )

    show_result("manual_predictions")

with eml_tab:
    st.write(
        "Upload an exported email. The V2 parser extracts its "
        "subject and plain-text/HTML body parts."
    )
    st.caption(
        "Attachments are excluded. Images are not read with OCR."
    )

    with st.form("eml_form"):
        uploaded_eml = st.file_uploader(
            "Choose an email file",
            type=["eml"],
        )
        analyze_eml = st.form_submit_button(
            "Analyze uploaded email",
            type="primary",
        )

    if analyze_eml:
        if uploaded_eml is None:
            st.warning("Choose an .eml file first.")
        elif uploaded_eml.size > 10 * 1024 * 1024:
            st.warning("Use an .eml file smaller than 10 MB.")
        else:
            try:
                subject, body, text = parser.parse_email(
                    uploaded_eml.getvalue()
                )

                records = pd.DataFrame([{
                    "subject": subject,
                    "text": text,
                }])

                save_result(
                    "eml_predictions",
                    records,
                    active_configurations,
                )

                with st.expander("Extracted email text"):
                    st.text(subject)
                    st.text(body[:20000])

                if "\ufffd" in text:
                    st.warning(
                        "Some characters could not be decoded. "
                        "The prediction may be affected."
                    )

            except Exception as exc:
                st.session_state.pop("eml_predictions", None)
                st.error(f"Could not process this email: {exc}")

    show_result("eml_predictions")

with batch_tab:
    st.write(
        "Upload a UTF-8 CSV with a `text` column, "
        "or with `subject` and `body` columns."
    )
    st.caption(
        f"Maximum {MAX_BATCH} emails per upload. "
        "With a text column, include the complete email text there. "
        "Subject is then used only as a display label."
    )

    template = pd.DataFrame([
        {
            "subject": "Project meeting",
            "body": "Please join the project review tomorrow.",
        },
        {
            "subject": "Claim your prize",
            "body": "Click now to claim your cash reward.",
        },
    ])

    st.download_button(
        "Download example CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="email_batch_template.csv",
        mime="text/csv",
    )

    with st.form("batch_form"):
        uploaded_csv = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
        )
        analyze_batch = st.form_submit_button(
            "Analyze batch",
            type="primary",
        )

    if analyze_batch:
        if uploaded_csv is None:
            st.warning("Choose a CSV file first.")
        elif uploaded_csv.size > 10 * 1024 * 1024:
            st.warning("Use a CSV file smaller than 10 MB.")
        else:
            try:
                batch = pd.read_csv(
                    uploaded_csv,
                    keep_default_na=False,
                    dtype=str,
                    encoding="utf-8-sig",
                )

                batch.columns = batch.columns.str.strip().str.lower()

                if batch.columns.duplicated().any():
                    raise ValueError("Duplicate column names found.")

                if not 1 <= len(batch) <= MAX_BATCH:
                    raise ValueError(
                        f"CSV must contain 1–{MAX_BATCH} emails."
                    )

                if "text" in batch.columns:
                    texts = batch["text"].map(parser.clean_text)
                    subjects = (
                        batch["subject"].map(parser.clean_text)
                        if "subject" in batch.columns
                        else pd.Series("", index=batch.index)
                    )

                elif {"subject", "body"}.issubset(batch.columns):
                    subjects = batch["subject"].map(parser.clean_text)
                    bodies = batch["body"].map(parser.clean_text)
                    texts = (
                        subjects + " " + bodies
                    ).map(parser.clean_text)

                else:
                    raise ValueError(
                        "CSV needs a text column, or both "
                        "subject and body columns."
                    )

                records = pd.DataFrame({
                    "subject": subjects,
                    "text": texts,
                }).reset_index(drop=True)

                save_result(
                    "batch_predictions",
                    records,
                    active_configurations,
                )

            except Exception as exc:
                st.session_state.pop("batch_predictions", None)
                st.error(f"Could not process the CSV: {exc}")

    show_result("batch_predictions")