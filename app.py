import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="Medical Cost Predictor",
    page_icon=":hospital:",
    layout="wide",
)

st.markdown(
    """
    <style>
    div.stFormSubmitButton > button {
        background-color: #16a34a;
        color: white;
        border: none;
        font-size: 18px;
        font-weight: 700;
    }
    div.stFormSubmitButton > button:hover {
        background-color: #15803d;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("insurance.csv")


@st.cache_resource
def train_model(data: pd.DataFrame):
    x = data.drop(columns=["charges"])
    y = data["charges"]

    categorical_cols = ["sex", "smoker", "region"]
    numeric_cols = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return pipeline, mae, r2


def main():
    # Header
    st.title("Medical Cost Prediction")
    st.markdown(
        "**Mini Project** | Chaudhary Marmikkumar Ashvinbhai | Enrollment: 251370680002"
    )
    st.divider()

    data = load_data()
    model, mae, r2 = train_model(data)

    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset Rows", f"{len(data):,}")
    col2.metric("Model MAE", f"${mae:,.2f}")
    col3.metric("R² Score", f"{r2:.3f}")
    col4.metric("Features Used", "6")

    st.divider()

    # Prediction form
    st.subheader("Predict Medical Insurance Cost")
    st.caption(
        "Use the inputs below to customize the candidate's features and predict their medical charges."
    )

    with st.form("predict_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            age = st.slider("Age", min_value=18, max_value=64, value=40, step=1)
            bmi = st.number_input(
                "BMI", min_value=15.0, max_value=52.0, value=25.0, step=0.1
            )
            children = st.slider("Number of Children", min_value=0, max_value=5, value=0, step=1)

        with col_b:
            sex = st.selectbox("Sex", options=["female", "male"])
            smoker = st.selectbox("Smoker", options=["no", "yes"])
            region = st.selectbox(
                "Region",
                options=["northeast", "southeast", "southwest", "northwest"],
            )

        st.markdown("")
        submit = st.form_submit_button(
            "Estimate Prediction", use_container_width=True, type="primary"
        )

    if submit:
        input_df = pd.DataFrame(
            [{
                "age": age,
                "sex": sex,
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": region,
            }]
        )
        predicted_cost = float(model.predict(input_df)[0])

        st.markdown(
            f"<div style='text-align:center; padding:24px; background:#ecfdf5;"
            f"border-radius:12px; border:2px solid #059669; margin-top:16px;'>"
            f"<p style='font-size:18px; color:#047857; margin:0;'>Estimated Annual Medical Charge</p>"
            f"<p style='font-size:52px; font-weight:900; color:#065f46; margin:8px 0 0 0;'>${predicted_cost:,.2f}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        with st.expander("Input Summary"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
