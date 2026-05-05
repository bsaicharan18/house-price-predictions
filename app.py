import streamlit as st
import pandas as pd
import pickle

# Title
st.title("House Price Prediction App 🏠")

# Inputs
LotArea = st.number_input("Enter Area (sq ft)")
BedroomAbvGr = st.number_input("BedroomAbvGr")
FullBath = st.number_input("FullBath")
OverallQual = st.number_input("OverallQual")
GrLivArea = st.number_input("GrLivArea")
GarageCars = st.number_input("GarageCars")
TotalBsmtSF = st.number_input("TotalBsmtSF")
YearBuilt = st.number_input("YearBuilt")
# Load model
model = pickle.load(open("model.pkl", "rb"))

# Prediction
if st.button("Predict Price"):
    result = model.predict([[OverallQual,GrLivArea,GarageCars,TotalBsmtSF,YearBuilt,FullBath,BedroomAbvGr,LotArea]])
    st.success(f"Predicted Price: {result[0]}")