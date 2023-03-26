# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:49:29 2023

@author: rbnro
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved model
loaded_model = pickle.load(open('F:/Study/Files/Semester Studies/8th Semester/0. Thesis/Deployment/trained_model.sav', 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Our Predictive Model', ['Cerebral Stroke Prediction'], icons=['person'], default_index=0)

# creating a function for prediction

def cspred(input_data):

    # Convert the dataframe row to a numpy array
    input_data_as_numpy_array = np.array(input_data.iloc[0])

    # Print the resulting numpy array
    #print(input_data_as_numpy_array)

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    c_s_prediction = loaded_model.predict(input_data_reshaped)
    print(c_s_prediction)

    if (c_s_prediction[0] == 0):
      return 'This person did not have a stroke'

    else:
      return 'This person had a stroke.'
  
    

    

def main():
    
    # giving a title
    st.title('Cerebral Stroke Prediction')
    
    # getting the input data from user
    
    gender = st.text_input("Enter gender (Male/Female/Other): ")
    age = st.text_input("Enter age: ")
    hypertension = st.text_input("Enter hypertension (1 for Yes, 0 for No): ")
    heart_disease = st.text_input("Enter heart disease (1 for Yes, 0 for No): ")
    ever_married = st.text_input("Enter ever married (Yes/No): ")
    work_type = st.text_input("Enter work type (Govt_job/Private/Self-employed/Children/Never_worked): ")
    residence_type = st.text_input("Enter residence type (Urban/Rural): ")
    avg_glucose_level = st.text_input("Enter average glucose level: ")
    bmi = st.text_input("Enter BMI: ")
    smoking_status = st.text_input("Enter smoking status (formerly smoked/never smoked/smokes/Unknown): ")
    
    # Create an empty pandas dataframe with the given columns
    input_data = pd.DataFrame(columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    
    # Add the user input to the pandas dataframe as a new row
    input_data.loc[0] = [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level, bmi, smoking_status]
    
    # Print the resulting dataframe
    #print(input_data)

    # convert categorical columns to numerical values
    input_data.replace({'ever_married':{'No':0,'Yes':1},'gender':{'Male':0,'Female':1,'Other':2},'work_type':{'Private':0,'Self-employed':1,'children':2,'Govt_job':3,'Never_worked':4},
                          'Residence_type':{'Rural':0,'Urban':1},'smoking_status':{'never smoked':0,'Unknown':1,'formerly smoked':2,'smokes':3}},inplace=True)

    # Print the resulting dataframe
    #print(input_data)

    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Predict'):
        diagnosis = cspred(input_data)
  
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
  
    
  
    
  
    
