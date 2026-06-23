
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("return_prediction_model.pkl")

st.title("Product Return Prediction System")

price = st.number_input("Price", min_value=0.0)
rating = st.slider("Rating", 0.0, 5.0, 3.0)

if st.button("Predict"):

    data = pd.DataFrame({
        "price":[price],
        "rating":[rating]
    })

    prediction = model.predict(data)[0]
    risk = model.predict_proba(data)[0][1]

    st.subheader("Result")

    st.write("Prediction:", prediction)
    st.write("Risk Score:", f"{risk*100:.2f}%")
