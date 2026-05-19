# Medical Cost Prediction (Mini Project)

**Student:** Chaudhary Marmikkumar Ashvinbhai  
**Enrollment:** 251370680002  
**Institution:** GTU — Data Science Mini Project  
**Topic:** Predicting medical insurance charges using machine learning

---

## Problem Statement

Healthcare costs are rising globally. Insurance companies and hospitals need accurate cost estimates to set premiums, plan budgets, and reduce financial risk. This project builds a **supervised regression** system that predicts annual medical insurance charges (`charges`) from demographic and lifestyle features.

## Dataset

| File | Description |
|------|-------------|
| [`insurance.csv`](insurance.csv) | Medical Cost Personal Dataset — **1,338 records**, 7 columns |
| [`medical_cost_dataset_generated.csv`](medical_cost_dataset_generated.csv) | Synthetic dataset (same schema) from the notebook pipeline |

**Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`  
**Target:** `charges` (USD)

**Source:** [Kaggle — Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Project Structure

```
├── medical_cost_prediction_complete.ipynb   # Full mini-project notebook (all PDF steps)
├── insurance.csv                            # Primary dataset
├── medical_cost_dataset_generated.csv       # Notebook-generated dataset
├── bulk_predictions_300_candidates.csv      # Sample bulk prediction output
├── app.py                                   # Streamlit web app
├── requirements.txt
└── README.md
```

## Methodology

1. **Problem identification** — healthcare cost estimation need  
2. **Data preprocessing** — null/duplicate checks, train–test split (80/20), one-hot encoding  
3. **EDA & visualization** — age, BMI, smoker, and region vs charges  
4. **Model** — Random Forest Regressor  
5. **Evaluation** — MAE, RMSE, R² on held-out test set  
6. **Deployment** — single + bulk predictions; Streamlit UI for live estimates  

### Model performance (test set)

| Metric | Value |
|--------|-------|
| MAE | ~$1,982 |
| RMSE | ~$2,517 |
| R² | ~0.93 |

## Tech Stack

Python 3.9+, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, Jupyter

## How to Run

```bash
cd 251370680002_Marmik_Medical_Cost_Prediction
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Notebook (full project):**

```bash
jupyter notebook medical_cost_prediction_complete.ipynb
```

**Web app:**

```bash
streamlit run app.py
```

## Sample Prediction

**Input:** age 39, female, BMI 29.4, 2 children, non-smoker, northeast  
**Output:** ~$14,623 annual charge

## References

1. Kaggle — Medical Cost Personal Dataset  
2. [scikit-learn](https://scikit-learn.org/)  
3. [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/)  
4. Medical Insurance Price Prediction Using Machine Learning (2024)  
5. Health Insurance Cost Prediction Using Regression Models (2022)

## Author

**Chaudhary Marmikkumar Ashvinbhai** — 251370680002
