import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide"
)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = pd.read_parquet("credit_card_data.parquet")

    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]

    legit_sample = legit.sample(n=len(fraud), random_state=2)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    return balanced_data


data = load_data()

X = data.drop(columns="Class")
y = data["Class"]

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=2
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# ---------------- HEADER ----------------
st.title("💳 Credit Card Fraud Detection System")
st.write("Professional Machine Learning Web Application by Zainul Abedeen")

st.markdown("---")

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)

col1.metric("Training Accuracy", f"{train_acc:.2f}")
col2.metric("Testing Accuracy", f"{test_acc:.2f}")
col3.metric("Total Samples", len(data))

st.markdown("---")

# ---------------- DEFAULT VALUES ----------------
legit_default = X[y == 0].iloc[0]
fraud_default = X[y == 1].iloc[0]

if "default_values" not in st.session_state:
    st.session_state.default_values = legit_default

colA, colB, colC = st.columns(3)

with colA:
    if st.button("Load Legitimate Sample"):
        st.session_state.default_values = legit_default

with colB:
    if st.button("Load Fraud Sample"):
        st.session_state.default_values = fraud_default

with colC:
    if st.button("Reset Values"):
        st.session_state.default_values = np.zeros(len(X.columns))

st.markdown("---")

# ---------------- INPUT FIELDS ----------------
cols = st.columns(4)
user_inputs = []

for i, feature in enumerate(X.columns):
    col = cols[i % 4]
    value = col.number_input(
        feature, value=float(st.session_state.default_values[i]), format="%.6f"
    )
    user_inputs.append(value)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("Analyze Transaction"):
    features = np.array(user_inputs).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    if prediction[0] == 0:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("🚨 Fraudulent Transaction")

    st.progress(float(probability))
    st.metric("Fraud Probability", f"{probability * 100:.2f}%")

st.markdown("---")
st.caption("Developed by Zainul | ML Project 🚀")
