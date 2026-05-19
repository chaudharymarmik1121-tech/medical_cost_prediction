import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


st.set_page_config(page_title="Medical Cost Predictor", page_icon=":hospital:", layout="wide")


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

    model = RandomForestRegressor(n_estimators=300, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
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
    st.title("Medical Cost Prediction")
    st.caption("Mini-project web app for estimating insurance charges.")

    data = load_data()
    model, mae, r2 = train_model(data)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset rows", f"{len(data):,}")
        st.metric("Model MAE", f"${mae:,.2f}")
    with col2:
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("Features used", "6")

    st.subheader("Predict Medical Insurance Cost")
    with st.form("predict_form"):
        age = st.slider("Age", min_value=18, max_value=64, value=30)
        sex = st.selectbox("Sex", options=["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=27.5, step=0.1)
        children = st.slider("Children", min_value=0, max_value=5, value=1)
        smoker = st.selectbox("Smoker", options=["yes", "no"])
        region = st.selectbox(
            "Region",
            options=["southwest", "southeast", "northwest", "northeast"],
        )
        submit = st.form_submit_button("Estimate Cost")

    if submit:
        input_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "bmi": bmi,
                    "children": children,
                    "smoker": smoker,
                    "region": region,
                }
            ]
        )
        predicted_cost = float(model.predict(input_df)[0])
        st.success(f"Estimated annual medical charge: ${predicted_cost:,.2f}")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
