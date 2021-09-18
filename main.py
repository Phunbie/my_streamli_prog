# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:55:33 2021

@author: FBI
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor



#load the data to prevent constantly reloading the data everytime we
#refresh the app
@st.cache
def load_data(nrows):
    energy = pd.read_csv('dataa/energydata_complete.csv', nrows=nrows)
    return energy

header = st.container()
dataset = st.container()
feature = st.container()
model_training = st.container()



st.sidebar.title('Sidebar')
energy = load_data(1000)
y = energy.Appliances

def model_sel(datas,mod,val):
    X = datas
    mod.fit(X,y)
    val = np.array(val).reshape(1,-1)
    predicted = mod.predict(val)
    return predicted
option = energy.columns.tolist()
visual_opp = energy.columns.tolist()
visual_opp.remove('date')
mod_features = energy.columns.tolist()
removel = ['Appliances','Tdewpoint','date']
mod_features = [i  for i in  mod_features if i not in removel]

features = st.sidebar.multiselect("Select the columns you want to view on the displayed Dataframe", option,default = option)
visualize = st.sidebar.selectbox('Select the columns you want to visualise', visual_opp)
visualize2 = st.sidebar.selectbox('select another column for comparision', visual_opp)
st.sidebar.header('Model selection options')
model_select = st.sidebar.selectbox('select the model you want',['Decision Tree','Linear Regression'])
input_feat = st.sidebar.multiselect('input features',mod_features,mod_features)

    



def sel_input(input_feat):
    for i in input_feat:
        min_v = (energy[i].min())
        max_v = (energy[i].max())
        resu = st.sidebar.slider(i,min_value=min_v,max_value=max_v)
        yield resu
in_option = list(sel_input(input_feat))

with header:
    st.title('My energy consumption webapp')
with dataset:
    st.header('A brief overview of the data(dataframe)')
    energy = load_data(1000)
    st.write(energy[features].head(10))
    st.header('Data visualisation(line graph)')
    st.line_chart(energy[[visualize,visualize2]][:50])
    if model_select == 'Decision Tree':
        predicted = model_sel(energy[mod_features],DecisionTreeRegressor(),in_option)
        
    else:
        predicted = model_sel(energy[mod_features],LinearRegression(),in_option)
    st.header('This is the predicted value')
    pre =  int(predicted)
    pre
        
    
    
    
    
    
    
#with model_training:
    
    