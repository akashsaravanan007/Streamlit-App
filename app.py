import os
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from transformers import pipeline

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# =============================
# SETTINGS
# =============================

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

OUTPUT_DIR = "synthetic_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================
# LOAD MODEL
# =============================

@st.cache_resource
def load_model():

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        framework="pt",
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.2,
        max_length=128
    )

    return generator


generator = load_model()


# =============================
# SESSION STATE INIT
# =============================

if "dataset" not in st.session_state:
    st.session_state.dataset = pd.DataFrame(columns=["text", "label"])


# =============================
# UI
# =============================

st.title("Synthetic Text Generation and Classification System")

tabs = st.tabs([
    "Generate Data",
    "Dataset Preview",
    "Train Classifier",
    "Visualization"
])


# =============================
# TAB 1 — GENERATE DATA
# =============================

with tabs[0]:

    st.header("Generate Synthetic Data")

    prompt = st.text_area("Enter Prompt")

    label = st.text_input("Enter Label (example: positive, negative, spam, ham)")

    num_samples = st.number_input("Number of Samples", 1, 500, 10)

    if st.button("Generate"):

        if prompt == "" or label == "":
            st.error("Enter both prompt and label")
            st.stop()

        progress_bar = st.progress(0)
        status = st.empty()

        new_data = []

        for i in range(num_samples):

            status.text(f"Generating {label} sample {i+1}/{num_samples}")

            output = generator(prompt)[0]["generated_text"]

            new_data.append({
                "text": output,
                "label": label
            })

            progress_bar.progress((i+1)/num_samples)

            time.sleep(0.05)

        new_df = pd.DataFrame(new_data)

        # append to existing dataset
        st.session_state.dataset = pd.concat(
            [st.session_state.dataset, new_df],
            ignore_index=True
        )

        st.success(f"{num_samples} samples added for label '{label}'")

        st.write("Total samples:", len(st.session_state.dataset))


# =============================
# TAB 2 — DATASET PREVIEW
# =============================

with tabs[1]:

    st.header("Dataset Preview")

    df = st.session_state.dataset

    if len(df) == 0:

        st.warning("No data generated yet")

    else:

        st.dataframe(df, use_container_width=True)

        st.write("Class distribution:")

        st.write(df["label"].value_counts())

        file_name = st.text_input("Enter File Name")

        if file_name:

            full_path = os.path.join(
                OUTPUT_DIR,
                file_name + ".csv"
            )

            if st.button("Save CSV"):

                df.to_csv(full_path, index=False)

                st.success(f"Saved to {full_path}")

            st.download_button(
                "Download CSV",
                df.to_csv(index=False),
                file_name=file_name + ".csv",
                mime="text/csv"
            )


# =============================
# TAB 3 — TRAIN CLASSIFIER
# =============================

with tabs[2]:

    st.header("Train Classifier")

    df = st.session_state.dataset

    if len(df) == 0:

        st.warning("Generate dataset first")

    else:

        num_classes = df["label"].nunique()

        st.write("Number of classes:", num_classes)

        if num_classes < 2:

            st.error("Need at least 2 classes to train classifier")

        else:

            if st.button("Train Model"):

                X = df["text"]
                y = df["label"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42
                )

                vectorizer = TfidfVectorizer()

                X_train_vec = vectorizer.fit_transform(X_train)
                X_test_vec = vectorizer.transform(X_test)

                model = LogisticRegression()

                model.fit(X_train_vec, y_train)

                predictions = model.predict(X_test_vec)

                accuracy = accuracy_score(y_test, predictions)

                st.session_state.model = model
                st.session_state.vectorizer = vectorizer
                st.session_state.y_test = y_test
                st.session_state.predictions = predictions
                st.session_state.accuracy = accuracy

                st.success("Training completed")

                st.write("Accuracy:", round(accuracy, 4))


# =============================
# TAB 4 — VISUALIZATION
# =============================

with tabs[3]:

    st.header("Visualization")

    if "accuracy" not in st.session_state:

        st.warning("Train model first")

    else:

        st.write("Accuracy:", round(st.session_state.accuracy, 4))

        cm = confusion_matrix(
            st.session_state.y_test,
            st.session_state.predictions
        )

        fig, ax = plt.subplots()

        disp = ConfusionMatrixDisplay(cm)

        disp.plot(ax=ax)

        st.pyplot(fig)