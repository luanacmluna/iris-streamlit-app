import pickle
import streamlit as st
import numpy as np

# Load the saved model and scaler
scaler = pickle.load(open('scaler_iris.pkl', 'rb'))
model = pickle.load(open('model_lr_iris.pkl', 'rb'))

# Streamlit app
st.title("Iris Species Predictor")

st.write("Enter sepal and petal dimensions to predict the species.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

if st.button("Predict"):
    # Transform input using the scaler
    input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Predict the species
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Species: {prediction[0]}")
