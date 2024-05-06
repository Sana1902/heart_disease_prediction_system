import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/saqla/OneDrive/Desktop/Sana/trained_model_project.sav','rb'))

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "The person does not have a heart disease."
    else:
        return "The person has heart disease."

def main():
    st.title("Heart Doctor: Heart Disease Predictor")
    
    age = st.text_input("Age:")
    sex = st.text_input("Sex:")
    cp = st.text_input("Cp value:")
    trestbps = st.text_input("Trestbps rate:")
    chol = st.text_input("Chol level:")
    fbs = st.text_input("Fbs value:")
    restecg = st.text_input("Restecg value:")
    thalach = st.text_input("Thalach level:")
    exang = st.text_input("Exang value:")
    oldpeak = st.text_input("Oldpeak value:")
    slope = st.text_input("Slope value:")
    ca = st.text_input("Ca value:")
    thal = st.text_input("Thal value:")
    
    diagnosis = ''
    
    if st.button("Predict"):
        diagnosis = heart_disease_prediction([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    
    st.success(diagnosis)
    
if __name__ == "__main__":
    main()
