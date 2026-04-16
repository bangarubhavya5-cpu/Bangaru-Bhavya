import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Purchase Prediction", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stButton>button {
        background-color: #00adb5;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Online Shopper Purchase Prediction")
st.markdown("### Predict customer purchase behavior using ML")

col1, col2, col3 = st.columns(3)

with col1:
    Administrative = st.number_input("Administrative", 0)
    Administrative_Duration = st.number_input("Administrative Duration", 0.0)
    Informational = st.number_input("Informational", 0)
    Informational_Duration = st.number_input("Informational Duration", 0.0)

with col2:
    ProductRelated = st.number_input("Product Related", 0)
    ProductRelated_Duration = st.number_input("Product Related Duration", 0.0)
    BounceRates = st.number_input("Bounce Rates", 0.0)
    ExitRates = st.number_input("Exit Rates", 0.0)

with col3:
    PageValues = st.number_input("Page Values", 0.0)
    SpecialDay = st.number_input("Special Day", 0.0)
    Month = st.selectbox("Month", ['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec'])
    VisitorType = st.selectbox("Visitor Type", ['Returning_Visitor','New_Visitor','Other'])
    Weekend = st.selectbox("Weekend", [True, False])

def preprocess_input():
    data = {
        'Administrative': Administrative,
        'Administrative_Duration': Administrative_Duration,
        'Informational': Informational,
        'Informational_Duration': Informational_Duration,
        'ProductRelated': ProductRelated,
        'ProductRelated_Duration': ProductRelated_Duration,
        'BounceRates': BounceRates,
        'ExitRates': ExitRates,
        'PageValues': PageValues,
        'SpecialDay': SpecialDay,
        'Month': Month,
        'VisitorType': VisitorType,
        'Weekend': Weekend
    }

    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=columns, fill_value=0)
    df_input = scaler.transform(df_input)

    return df_input

st.markdown("<br>", unsafe_allow_html=True)
center = st.columns([1,1,1])[1]

with center:
    predict_btn = st.button("Predict Purchase")

if predict_btn:
    input_data = preprocess_input()
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prob < 0.4:
        level = "Low Probability"
    elif prob < 0.7:
        level = "Medium Probability"
    else:
        level = "High Probability"

    if prob >= 0.5:
        st.success("Customer WILL Purchase")
    else:
        st.error("Customer will NOT Purchase")

    st.info(f"Confidence Score: {prob*100:.1f}%")
    st.progress(int(prob * 100))
    st.write(f"Prediction Strength: **{level}**")

    if prob < 0.4:
        st.warning("Low chance of purchase → Consider marketing strategies.")
    elif prob < 0.7:
        st.warning("Moderate chance → Improve engagement.")
    else:
        st.success("High chance → Target this customer!")