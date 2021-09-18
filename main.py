# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:55:33 2021

@author: FBI
"""
import streamlit as st
import pandas as pd
import numpy as np
#import plotly.figure_factory as ff
#import matplotlib.pyplot as plt
#import seaborn as sns




#load the data to prevent constantly reloading the data every
#refresh the app
@st.cache
def load_data(nrows):
    energy = pd.read_csv('dataa/energydata_complete.csv', nrows=nrows)
    return energy

header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()



st.sidebar.header('this is where tuning occur')
energy = load_data(1000)
option = energy.columns.tolist()
features = st.sidebar.multiselect("You can select the columns you want", option,default = option)
selected_col = []
for i in features:
    selected_col.append(i)
min_value_app=int(energy.Appliances.min())
max_value_app=int(energy.Appliances.max())
appl_num = st.sidebar.slider('Number of dataset',min_value=min_value_app,max_value=max_value_app)
with header:
    st.title('My energy consumption webapp')
with dataset:
    st.header('A brief overview of the data')
    energy = load_data(500)
    st.write(energy[selected_col].head(10))
    st.line_chart(energy[['Appliances','Visibility']][:50])
    

with feature:
    st.header('we will display selected features')
    
    
    
    
    
    
#with model_training:
    
    