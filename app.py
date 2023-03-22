import streamlit as st
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open('model.sav', 'rb'))
# Load the trained model

# Define a function to preprocess user input
def preprocess_input(age, sex, bmi, children, smoker, region):
    
    # Preprocess sex
    sex = 1 if sex == "Male" else 0
    
    # Preprocess smoker
    smoker = 1 if smoker == "yes" else 0
    
    # Preprocess region
    regions = ['northeast', 'northwest', 'southeast', 'southwest']
    region_dict = {key: 0 for key in regions}
    region_dict[region] = 1
    
    # Combine all the preprocessed values into a numpy array
    input_array = np.array([[age, sex, bmi, children, smoker, 
                             region_dict['northeast'], region_dict['northwest'], 
                             region_dict['southeast'], region_dict['southwest']]])
    
    
    return input_array
 

def main():
    # Set up the app interface
    st.title("Health Insurance Cost Predictor")
    st.write("This app predicts the health insurance cost based on user inputs.")

    # Get user inputs
    age = st.number_input("Age:")
    sex = st.selectbox("Sex:", options=["Male", "Female"])
    bmi = st.number_input("BMI:")
    children = st.number_input("Number of children:")
    smoker = st.selectbox("Smoker:", options=["yes", "no"])
    region = st.selectbox("Region:", options=["northeast", "northwest", "southeast", "southwest"])

    # Preprocess user inputs
    input_df = preprocess_input(age, sex, bmi, children, smoker, region)

    # Make predictions and display the results
    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        st.write(f"The predicted cost of health insurance is {prediction:.2f} dollars.")


if __name__ == "__main__":
    main()
