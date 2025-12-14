import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD MODEL ----------------
with open("model/studentPerformanceRidge.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/StudentPerformanceScaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🎓 Student Math Score Prediction")

# ---------------- MAPPINGS ----------------
gender_map = {"female": 0, "male": 1}

race_map = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4
}

parent_edu_map = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}


test_prep_map = {"none": 0, "completed": 1}

# ---------------- INPUTS ----------------
gender = st.selectbox("Gender", gender_map.keys())
if gender == "female":
    gender_female = 1
    gender_male = 0
else:
    gender_female = 0
    gender_male = 1

race = st.selectbox("Race / Ethnicity", race_map.keys())
parent_edu = st.selectbox("Parental Education", parent_edu_map.keys())
test_prep = st.selectbox("Test Preparation Course", test_prep_map.keys())

Writting_score = st.number_input("Writting Score", 0, 100, 50)
reading_score = st.number_input("Reading Score", 0, 100, 50)

# ---------------- PREDICTION ----------------
if st.button("Predict Writing Score"):

    input_data = np.array([[
        race_map[race],
        parent_edu_map[parent_edu],
        test_prep_map[test_prep],
        reading_score,
        Writting_score,
        gender_female,
        gender_male,
    ]])

    # scale numeric features
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    st.success(f"📊 Predicted Math Score: {prediction[0]:.2f}")
