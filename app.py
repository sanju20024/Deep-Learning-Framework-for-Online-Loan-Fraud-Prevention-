import gdown
import os

# Download dataset
if not os.path.exists("dataset.csv"):
    gdown.download("https://drive.google.com/uc?id=1jVZgVKycfybIT2P-MYkF-a_5RpH_qmUG", "dataset.csv", quiet=False)

# Download KNN model
if not os.path.exists("knn_classifier.joblib"):
    gdown.download("https://drive.google.com/uc?id=1tYool1WWHSr5iHbSAOaxGnrn6MtBWi-o", "knn_classifier.joblib", quiet=False)

# Download MLP model
if not os.path.exists("mlp_classifier.joblib"):
    gdown.download("https://drive.google.com/uc?id=1KaMDKhp-GXRDj0NbO-tdnqLcJtFBTs3t", "mlp_classifier.joblib", quiet=False)

# Download test data
if not os.path.exists("test_data.csv"):
    gdown.download("https://drive.google.com/uc?id=1JxD40Q727X0nLkUqp1y-ZOQB91p0e4Mq", "test_data.csv", quiet=False)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib

st.set_page_config(layout="wide")
st.title("🔐 Fraud Detection using Neural Network Architecture")

# Initialize session_state variables
for key in ["dataset", "X", "y", "x_train", "x_test", "y_train", "y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None

le = LabelEncoder()

uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type=["csv"])

if uploaded_file is not None:
    st.session_state["dataset"] = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded")
    st.dataframe(st.session_state["dataset"].head())

# ---- Preprocessing Button ----
if st.button("🔄 Preprocess Dataset"):
    try:
        data = st.session_state["dataset"]
        data['type'] = le.fit_transform(data['type'])
        data['nameOrig'] = le.fit_transform(data['nameOrig'])
        data['nameDest'] = le.fit_transform(data['nameDest'])
        data.fillna(0, inplace=True)

        st.session_state["X"] = data.drop('isFraud', axis=1)
        st.session_state["y"] = data['isFraud']

        st.success("✅ Preprocessing completed")
        st.subheader("Count Plot before SMOTE")
        fig, ax = plt.subplots()
        sns.countplot(x=st.session_state["y"], palette="Set2", ax=ax)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during preprocessing: {e}")

# ---- Apply SMOTE Button ----
if st.button("🧪 Apply SMOTE & Split"):
    X, y = st.session_state["X"], st.session_state["y"]
    if X is not None and y is not None:
        try:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            st.session_state["X"], st.session_state["y"] = X, y
            st.session_state["x_train"], st.session_state["x_test"] = x_train, x_test
            st.session_state["y_train"], st.session_state["y_test"] = y_train, y_test

            st.success("✅ SMOTE applied and data split done")
            st.info(f"Training size: {x_train.shape[0]}, Testing size: {x_test.shape[0]}")

            st.subheader("Count Plot after SMOTE")
            fig, ax = plt.subplots()
            sns.countplot(x=y, palette="Set2", ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SMOTE error: {e}")
    else:
        st.warning("⚠️ Please run preprocessing first.")

# ---- Train KNN Button ----
if st.button("🤖 Train KNN Classifier"):
    try:
        x_train = st.session_state["x_train"]
        y_train = st.session_state["y_train"]
        x_test = st.session_state["x_test"]
        y_test = st.session_state["y_test"]

        model = KNeighborsClassifier(n_neighbors=10)
        model.fit(x_train, y_train)
        joblib.dump(model, "knn_classifier.joblib")

        pred = model.predict(x_test)
        st.subheader("📊 KNN Classifier Metrics")
        st.write(f"Precision: {precision_score(y_test, pred, average='macro') * 100:.2f}%")
        st.write(f"Recall: {recall_score(y_test, pred, average='macro') * 100:.2f}%")
        st.write(f"F1 Score: {f1_score(y_test, pred, average='macro') * 100:.2f}%")
        st.write(f"Accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
        st.pyplot(fig)

    except Exception:
        st.warning("⚠️ Please complete SMOTE + split before training.")

# ---- Train MLP Button ----
if st.button("🧠 Train MLP Classifier"):
    try:
        x_train = st.session_state["x_train"]
        y_train = st.session_state["y_train"]
        x_test = st.session_state["x_test"]
        y_test = st.session_state["y_test"]

        model = MLPClassifier(max_iter=500)
        model.fit(x_train, y_train)
        joblib.dump(model, "mlp_classifier.joblib")

        pred = model.predict(x_test)
        st.subheader("📊 MLP Classifier Metrics")
        st.write(f"Precision: {precision_score(y_test, pred, average='macro') * 100:.2f}%")
        st.write(f"Recall: {recall_score(y_test, pred, average='macro') * 100:.2f}%")
        st.write(f"F1 Score: {f1_score(y_test, pred, average='macro') * 100:.2f}%")
        st.write(f"Accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")

        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", ax=ax)
        st.pyplot(fig)

    except Exception:
        st.warning("⚠️ Please complete SMOTE + split before training.")
