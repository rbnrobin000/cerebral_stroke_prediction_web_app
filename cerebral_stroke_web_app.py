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
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Predictive System', ['Cerebral Stroke'], icons=['person'], default_index=0)
    

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
      return 'This person do not have any risk of cerebral stroke.'

    else:
      return 'This person has a risk of cerebral stroke.'
  
    

    

def main():
    
    # Cerebral Stroke Page
    if (selected == 'Cerebral Stroke'):
        
        # giving a title
        #st.title('Cerebral Stroke Prediction')
        st.markdown("<h1 style='text-align: center; color: grey;'>Cerebral Stroke Prediction</h1>", unsafe_allow_html=True)
        
        
        # getting the input data from the user
        col1, col2 = st.columns(2)
    
        with col1:
            gender = st.text_input("Enter gender (Male/Female/Other): ")
        with col1:
            age = st.text_input("Enter age: ")
        with col1:
            hypertension = st.text_input("Enter hypertension (1 for Yes, 0 for No): ")
        with col1:
            heart_disease = st.text_input("Enter heart disease (1 for Yes, 0 for No): ")
        with col1:
            ever_married = st.text_input("Enter ever married (Yes/No): ")
        with col2:
            work_type = st.text_input("Enter work type (Govt_job/Private/Self-employed/Children/Never_worked): ")
        with col2:
            residence_type = st.text_input("Enter residence type (Urban/Rural): ")
        with col2:
            avg_glucose_level = st.text_input("Enter average glucose level: ")
        with col2:
            bmi = st.text_input("Enter BMI: ")
        with col2:
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
  
    
  
    

    
