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
import seaborn as sns
import matplotlib.pyplot as plt


#loads the data to prevent constantly reloading the data everytime we
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

def model_sel(datas,mod,val): #function for model prediction
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
visualize2 = st.sidebar.selectbox('select another column for comparison', visual_opp)
st.sidebar.header('Model selection options')
model_select = st.sidebar.selectbox('select the machine learning model you want for prediction',['Decision Tree','Linear Regression'])
input_feat = st.sidebar.multiselect('input features',mod_features,mod_features[:10])

    



def sel_input(input_feat):
    for i in input_feat:
        min_v = (energy[i].min())
        max_v = (energy[i].max())
        resu = st.sidebar.slider(i,min_value=min_v,max_value=max_v)
        yield resu
in_option = list(sel_input(input_feat))

with header:
    st.title('Energy consumption prediction webapp')
with dataset:
    st.header('A brief overview of the data(dataframe)')
    energy = load_data(1000)
    st.write(energy[features].head(10))
    st.header('Data visualisation(line graph)')
    st.line_chart(energy[[visualize,visualize2]][:50])
    st.header('Data visualisation(regression plot)')
    fig2, ax = plt.subplots(figsize=(8,6))
    X_reg = visualize
    y_reg = visualize2
    sns.regplot(X_reg,y_reg,data=energy,ax=ax)
    st.write(fig2)
    st.header('Data visualisation(correlation heat map)')
    fig, ax = plt.subplots(figsize=(8,6))
    corre = energy[input_feat]
    cor = corre.corr()
    sns.heatmap(cor,ax=ax,annot=True)
    st.write(fig)
    if model_select == 'Decision Tree':
        predicted = model_sel(energy[input_feat],DecisionTreeRegressor(),in_option)
        
    else:
        predicted = model_sel(energy[input_feat],LinearRegression(),in_option)
  
    st.header('The predicted  energy use is {} watt per hour'.format(round(int(predicted))))    
    
    
    
    
    
    
#with model_training:
    
    