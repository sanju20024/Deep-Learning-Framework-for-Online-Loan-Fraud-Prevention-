# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

st.set_page_config(layout="wide")
st.title("🔐 Fraud Detection using Advanced Neural Network Architectures")

# Globals
dataset = None
X, y = None, None
x_train, x_test, y_train, y_test = None, None, None, None
le = LabelEncoder()
accuracy, precision, recall, fscore = [], [], [], []

# Step 1: Upload dataset
uploaded_file = st.file_uploader("📂 Upload your dataset (.csv)", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully")
    st.dataframe(dataset.head())

    # Step 2: Preprocess
    if st.button("🔄 Preprocess Dataset"):
        dataset['type'] = le.fit_transform(dataset['type'])
        dataset['nameOrig'] = le.fit_transform(dataset['nameOrig'])
        dataset['nameDest'] = le.fit_transform(dataset['nameDest'])
        dataset.fillna(0, inplace=True)

        X = dataset.drop('isFraud', axis=1)
        y = dataset['isFraud']

        st.subheader("Count Plot before SMOTE")
        fig, ax = plt.subplots()
        sns.countplot(x=y, palette="Set2", ax=ax)
        st.pyplot(fig)

    # Step 3: SMOTE & Split
    if st.button("🧪 Apply SMOTE & Split Data"):
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.success(f"Training records: {x_train.shape[0]}, Testing records: {x_test.shape[0]}")
        
        st.subheader("Count Plot after SMOTE")
        fig, ax = plt.subplots()
        sns.countplot(x=y, palette="Set2", ax=ax)
        st.pyplot(fig)

    # Step 4: Train & Evaluate KNN
    if st.button("🤖 Train KNN Classifier"):
        knn = KNeighborsClassifier(n_neighbors=10, leaf_size=30, metric='minkowski')
        knn.fit(x_train, y_train)
        joblib.dump(knn, "knn_classifier.joblib")

        pred = knn.predict(x_test)
        p = precision_score(y_test, pred, average='macro') * 100
        r = recall_score(y_test, pred, average='macro') * 100
        f = f1_score(y_test, pred, average='macro') * 100
        a = accuracy_score(y_test, pred) * 100
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(a)

        st.subheader("KNN Classifier Metrics")
        st.write(f"**Precision:** {p:.2f}%")
        st.write(f"**Recall:** {r:.2f}%")
        st.write(f"**F1-Score:** {f:.2f}%")
        st.write(f"**Accuracy:** {a:.2f}%")

        cm = confusion_matrix(y_test, pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

    # Step 5: Train & Evaluate MLP
    if st.button("🧠 Train MLP Classifier"):
        mlp = MLPClassifier(max_iter=500)
        mlp.fit(x_train, y_train)
        joblib.dump(mlp, "mlp_classifier.joblib")

        pred = mlp.predict(x_test)
        p = precision_score(y_test, pred, average='macro') * 100
        r = recall_score(y_test, pred, average='macro') * 100
        f = f1_score(y_test, pred, average='macro') * 100
        a = accuracy_score(y_test, pred) * 100
        precision.append(p)
        recall.append(r)
        fscore.append(f)
        accuracy.append(a)

        st.subheader("MLP Classifier Metrics")
        st.write(f"**Precision:** {p:.2f}%")
        st.write(f"**Recall:** {r:.2f}%")
        st.write(f"**F1-Score:** {f:.2f}%")
        st.write(f"**Accuracy:** {a:.2f}%")

        cm = confusion_matrix(y_test, pred)
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        st.pyplot(fig)

    # Step 6: Comparison Graph
    if st.button("📊 Show Comparison Graph"):
        metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
        models = ['KNN', 'MLP']
        values = [
            [precision[0], recall[0], fscore[0], accuracy[0]],
            [precision[-1], recall[-1], fscore[-1], accuracy[-1]]
        ]

        df = pd.DataFrame(values, columns=metrics, index=models)
        st.bar_chart(df.T)

    # Step 7: Upload new test file for prediction
    st.subheader("🔍 Predict from New Data")
    test_file = st.file_uploader("Upload new CSV for prediction", key="test")
    if test_file is not None:
        test_data = pd.read_csv(test_file)
        test_data['type'] = le.fit_transform(test_data['type'])
        test_data['nameOrig'] = le.fit_transform(test_data['nameOrig'])
        test_data['nameDest'] = le.fit_transform(test_data['nameDest'])

        model_choice = st.selectbox("Select model for prediction", ("KNN", "MLP"))
        if st.button("🔮 Predict"):
            model_file = "knn_classifier.joblib" if model_choice == "KNN" else "mlp_classifier.joblib"
            model = joblib.load(model_file)
            pred = model.predict(test_data)
            test_data["Prediction"] = ["Fraud" if p == 1 else "Genuine" for p in pred]
            st.write(test_data)
