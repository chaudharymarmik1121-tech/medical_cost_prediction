# 🏥 Medical Cost Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=chaudharymarmik1121-tech/medical_cost_prediction&branch=main&mainModule=app.py)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-Educational-green)

> **GTU Post Graduate Diploma in Data Science — Mini Project**  
> **Student:** Chaudhary Marmikkumar Ashvinbhai | **Enrollment:** 251370680002

---

## 🚀 Live Demo

Click the button below to run the live prediction app:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=chaudharymarmik1121-tech/medical_cost_prediction&branch=main&mainModule=app.py)

**Or deploy it yourself in 1 click → [share.streamlit.io](https://share.streamlit.io)**  
Select this repo → set main file as `app.py` → Deploy ✅

---

## 📌 Problem Statement

Healthcare costs are rising globally. Insurance companies need accurate estimates of annual medical charges to set fair premiums and manage financial risk.

**Goal:** Predict annual medical insurance `charges` (USD) from patient demographics and lifestyle inputs using supervised machine learning.

---

## 📊 Dataset

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Age of the beneficiary (18–64) |
| `sex` | Categorical | Gender (male / female) |
| `bmi` | Numeric | Body Mass Index (15–53) |
| `children` | Numeric | Number of dependents (0–5) |
| `smoker` | Categorical | Smoking status (yes / no) |
| `region` | Categorical | US region (northeast / southeast / southwest / northwest) |
| `charges` | **Target** | Annual insurance charges in USD |

- **Records:** 1,338
- **Source:** [Kaggle — Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

## ⚙️ Methodology

```
Data → EDA → Preprocessing → Model Training → Evaluation → Prediction
```

1. **EDA** — Age, BMI, smoking status vs charges analysis
2. **Preprocessing** — One-Hot Encoding, 80/20 train-test split
3. **Model** — Random Forest Regressor (350 trees)
4. **Evaluation** — MAE, RMSE, R² on hold-out test set
5. **Deployment** — Streamlit web app for live predictions

---

## 📈 Model Results

| Metric | Value |
|--------|-------|
| **MAE** | $1,982.25 |
| **RMSE** | $2,517.21 |
| **R² Score** | 0.9331 (93.3%) |

> Key insight: Smokers pay **~4x more** ($32,050 avg) than non-smokers ($8,434 avg)

---

## 🧪 Sample Prediction

| Input | Value |
|-------|-------|
| Age | 39 |
| Sex | Female |
| BMI | 29.4 |
| Children | 2 |
| Smoker | No |
| Region | Northeast |
| **Predicted Charge** | **~$14,623/year** |

---

## 🗂️ Project Structure

```
medical_cost_prediction/
├── app.py                                    # Streamlit web application
├── medical_cost_prediction_complete.ipynb    # Full analysis notebook
├── insurance.csv                             # Dataset (1,338 records)
├── requirements.txt                          # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.9+ | Core language |
| pandas & NumPy | Data handling |
| scikit-learn | ML pipeline, Random Forest |
| matplotlib & seaborn | Visualizations |
| Streamlit | Web application |
| Jupyter Notebook | Analysis & report |

---

## 💻 Run Locally

```bash
git clone https://github.com/chaudharymarmik1121-tech/medical_cost_prediction.git
cd medical_cost_prediction
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌍 Deploy Online (Free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub (`chaudharymarmik1121-tech`)
3. Click **New app** → Select this repository
4. Set **Main file:** `app.py`
5. Click **Deploy** → Live URL generated instantly ✅

---

## 📚 References

1. Kaggle — Medical Cost Personal Dataset
2. [scikit-learn](https://scikit-learn.org/) — Random Forest, Pipelines
3. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
4. [Streamlit Documentation](https://docs.streamlit.io/)
5. Medical Insurance Price Prediction Using ML (2024)

---

## 👤 Author

**Chaudhary Marmikkumar Ashvinbhai**  
Enrollment: 251370680002  
Gujarat Technological University (GTU) — PGDDS Mini Project
