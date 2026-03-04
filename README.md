# 💳 Credit Card Fraud Detection System

A professional Machine Learning web application built using **Logistic Regression** and **Streamlit** to detect fraudulent credit card transactions in real time.

This project demonstrates a clean and production-style ML workflow with an interactive prediction interface.

---

## 🚀 Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem.

This application:

* Balances the dataset using **undersampling**
* Trains a **Logistic Regression model**
* Evaluates performance using accuracy metrics
* Provides an interactive UI for real-time fraud prediction
* Displays fraud probability dynamically

The system is built with simplicity, clarity, and portfolio-level presentation in mind.

---

## 🧠 Machine Learning Workflow

1. Load dataset
2. Separate legitimate and fraudulent transactions
3. Apply undersampling to balance classes
4. Split dataset into training and testing sets
5. Train Logistic Regression model
6. Evaluate model performance
7. Deploy using Streamlit

---

## 📊 Application Features

✔ Clean Professional UI
✔ Training & Testing Accuracy Display
✔ Balanced Dataset Handling
✔ Default Legitimate Transaction Loader
✔ Default Fraud Transaction Loader
✔ Reset Feature Values Option
✔ Real-time Fraud Probability Indicator
✔ Interactive Prediction Interface

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── app.py
├── creditcard.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation Guide (Local Setup)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have `requirements.txt`, install manually:

```bash
pip install streamlit pandas numpy scikit-learn
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your browser.

---

## Application

![img]("https://github.com/zainulabedeen589/Detecting-Credit-Card-Fraud-with-Streamlit/blob/main/app.png")

## ☁️ Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Select your repository
3. Set main file as:

```text
app.py

```

Your application will be live within minutes.

---
## Note: Data is available on Kaggle.

## 🎯 Future Improvements

* Model persistence using joblib (.pkl)
* ROC-AUC evaluation
* Feature importance visualization
* REST API integration
* Docker containerization
* CI/CD pipeline setup

---

## 👨‍💻 Developer

## **Zainul Abedeen**

### [GitHub](https://github.com/zainulabedeen589)

### [LinkedIn](https://linkedin.com/in/zainulabedeen589)

---

## ⭐ Support

If you found this project helpful, consider giving it a star on GitHub.
