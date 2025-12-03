import streamlit as st
from sklearn.datasets import load_breast_cancer

from predict import predict

cancer = load_breast_cancer()
feature_names = cancer.feature_names
target_names = cancer.target_names

sample_data = {
    "mean radius":17.99,"mean texture":10.38,"mean perimeter":122.8,"mean area":1001,
    "mean smoothness":0.1184,"mean compactness":0.2776,"mean concavity":0.3001,
    "mean concave points":0.1471,"mean symmetry":0.2419,"mean fractal dimension":0.07871,
    "radius error":1.095,"texture error":0.9053,"perimeter error":8.589,"area error":153.4,
    "smoothness error":0.006399,"compactness error":0.04904,"concavity error":0.05373,
    "concave points error":0.01587,"symmetry error":0.03003,"fractal dimension error":0.006193,
    "worst radius":25.38,"worst texture":17.33,"worst perimeter":184.6,"worst area":2019,
    "worst smoothness":0.1622,"worst compactness":0.6656,"worst concavity":0.7119,
    "worst concave points":0.2654,"worst symmetry":0.4601,"worst fractal dimension":0.1189
}

st.title("Breast Cancer Prediction App")
st.write("Enter the tumor's features below to predict whether it is good or bad.")

inputs = []
for name in feature_names:
    default_value = float(sample_data.get(name, 1.0))
    value = st.number_input(name, min_value=0.0, max_value=10000.0, value=default_value, step=0.01)
    inputs.append(value)

if st.button("Predict"):
    result = predict(inputs)
    diagnosis = target_names[result]
    if diagnosis == "malignant":
        st.error(f"Prediction: {diagnosis.upper()} — Cancer detected")
    else:
        st.success(f"Prediction: {diagnosis.upper()} — Likely benign")
