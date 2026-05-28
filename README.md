# 🏥 Medical Insurance Cost Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://medicalcostprediction-whigsz4mgzy8rrtyuzqyvs.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

> **GTU Post Graduate Diploma in Data Science — Mini Project**
> **Student:** Chaudhary Marmikkumar Ashvinbhai | **Enrollment:** 251370680002
> **Guide:** Komal Prajapati

---

## 🚀 Live Demo

🔗 **https://medicalcostprediction-whigsz4mgzy8rrtyuzqyvs.streamlit.app/**

---

## 📌 Problem Statement

Healthcare costs are rising globally. Insurance companies need accurate estimates of annual medical charges to set fair premiums and manage risk.

**Goal:** Predict annual medical insurance `charges` (USD) from 13 patient health and demographic parameters using supervised machine learning.

---

## 📊 Dataset — Extended (10,000 Records)

| Feature              | Type        | Description                                               |
|----------------------|-------------|-----------------------------------------------------------|
| age                  | Numeric     | Patient age (18–64)                                       |
| sex                  | Categorical | Biological sex (male / female)                            |
| bmi                  | Numeric     | Body Mass Index (15–52)                                   |
| children             | Numeric     | Number of dependants (0–5)                                |
| smoker               | Categorical | Smoking status (yes / no)                                 |
| region               | Categorical | Coverage region (northeast / southeast / southwest / northwest) |
| diabetes             | Binary      | Diabetes diagnosis (0 = No, 1 = Yes)                      |
| blood_pressure       | Binary      | Hypertension / High BP (0 = No, 1 = Yes)                  |
| asthma               | Binary      | Asthma / respiratory condition (0 = No, 1 = Yes)          |
| skin_allergy         | Binary      | Skin condition / allergy (0 = No, 1 = Yes)                |
| cancer_history       | Binary      | Past or active cancer diagnosis (0 = No, 1 = Yes)         |
| chronic_infection    | Binary      | Recurring or long-term infection (0 = No, 1 = Yes)        |
| allergies            | Binary      | Food, drug, or environmental allergies (0 = No, 1 = Yes)  |
| **charges**          | **Target**  | **Annual insurance charges in USD**                       |

- **Records:** 10,000 synthetic patient records
- **Features:** 13 health & demographic parameters
- **Target:** Annual medical cost (USD)

---

## ⚙️ Methodology

```
Data Generation → EDA → Preprocessing → Model Training → Evaluation → Deployment
```

1. **Dataset** — Synthetic 10K records with medically realistic cost premiums per condition
2. **EDA** — Distribution analysis, condition impact, smoking vs cost analysis
3. **Preprocessing** — One-Hot Encoding for categoricals, 80/20 train-test split
4. **Model** — Random Forest Regressor (200 trees, n_jobs=-1)
5. **Evaluation** — MAE, R² score on held-out test set
6. **Deployment** — Streamlit web app with clinical-style patient intake UI

---

## 📈 Model Performance

| Metric       | Value      |
|--------------|------------|
| **R² Score** | ~0.97      |
| **MAE**      | ~$1,200    |
| **Trees**    | 200        |
| **Train Set**| 8,000 records |
| **Test Set** | 2,000 records |

---

## 💡 Key Insights

| Condition         | Estimated Annual Premium Impact |
|-------------------|---------------------------------|
| 🚬 Smoker         | +$23,800                        |
| 🎗️ Cancer History | +$13,000                        |
| 🩸 Diabetes       | +$5,000                         |
| ❤️ Hypertension   | +$3,500                         |
| 🌬️ Asthma         | +$3,000                         |
| 🦠 Chronic Infection | +$2,500                      |
| 🧴 Skin Allergy   | +$1,200                         |
| 🤧 Allergies      | +$800                           |

---

## 🗂️ Project Structure

```
251370680002_Marmik_Medical_Cost_Prediction/
├── app.py                                 # Streamlit web application (13 features)
├── medical_cost_prediction_complete.ipynb # Full ML analysis notebook
├── medical_cost_dataset_extended.csv      # Extended dataset (10,000 records, 13 features)
├── insurance.csv                          # Original dataset (1,338 records, 6 features)
├── requirements.txt                       # Python dependencies
├── runtime.txt                            # Python version for Streamlit Cloud
└── README.md
```

---

## 🛠️ Tech Stack

| Tool            | Purpose                        |
|-----------------|--------------------------------|
| Python 3.11     | Core language                  |
| pandas & NumPy  | Data handling & generation     |
| scikit-learn    | ML pipeline, Random Forest     |
| Streamlit       | Clinical-style web application |
| Jupyter Notebook| Analysis & documentation       |

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

## 👤 Author

**Chaudhary Marmikkumar Ashvinbhai**
Enrollment: 251370680002
Gujarat Technological University (GTU) — PGDDS Mini Project 2025-26
