import pandas as pd
import numpy as np
import streamlit as st
# from prediction import predict
st.title('Portuges Bank')
st.markdown('Model to predict the Portuges bank is customer buy the product or not')

st.header("House Features")


col1,col2,col3 = st.columns(3)
with col1:
    age      = st.slider('Age',17,69,35)
    duration =st.slider('Duration',0.0,4918.0,258.107493)
    campaign =st.slider('Campaign',1,56,2)
    pdays    = st.slider('pdays',0,999,181)
    cons_price_idx = st.slider('cons_price_idx',92.201000,94.767000,93.580967)
    cons_conf_idx = st.slider('cons_conf_idx',-50.800000,-26.900000,-40.549267)
    nr_employed	  = st.slider('nr_employed',4963.600000,5228.100000,5168.608770)
    euribor3m =st.slider('euribor3m ',0.634000,5.045000,3.652601)
with col2:
    job	 = st.radio('Job: ', [0,1, 2, 3, 4, 5, 6,7,8,9,10,11],horizontal=True)
    marital = st.radio('Marital',[0,1,2,3],horizontal=True)
    education = st.radio('education',[0,1, 2, 3, 4, 5, 6,7],horizontal=True)
    default =st.radio('default ',[0,1],horizontal=True)
    housing =st.radio('housing',[0,1,2],horizontal=True)
    loan	 = st.radio('loan : ',[0,1,2],horizontal=True)
with col3:
    contact = st.radio('contact ',[0,1],horizontal=True)
    month = st.radio('Month ',[0,1, 2, 3, 4, 5, 6,7,8,9],horizontal=True)
    day_of_week	 = st.radio('day_of_week ',[0,1, 2, 3, 4],horizontal=True)
    previous = st.radio('previous ',[0,1, 2, 3, 4, 5, 6,7],horizontal=True)
    poutcome = st.radio('poutcome ',[0,1,2],horizontal=True)
    emp_var_rate   = st.radio('emp.var.rate ',[1.4,-1.8,1.1,-0.1,-2.9,-3.4,-1.7,-1.1,-3.0,-0.2],horizontal=True)

if st.button('Risk Prediction'):
   # Read the dataset
    df = pd.read_csv('clean_data.csv')
    df = df.drop(['y'],axis=1)

   # Load the model
    import pickle
    # model = pickle.load(open('RFC.pkl', 'rb'))
    with open('RFC.pkl', 'rb') as file:
        model = pickle.load(file)
    # prediction = model.predict([[age,	job,	marital,	education,	default,	housing,	loan,	contact,	month,	day_of_week	,duration	
    #                             ,campaign	,pdays	,previous	,poutcome	,	cons_price_idx,	cons_conf_idx	,
    #                             euribor3m,	nr_employed]])
    prediction = ([[age,	job,	marital,	education,	default,	housing,	loan,	contact,	month,	day_of_week	,duration	
                    ,campaign	,pdays	,previous	,poutcome 	,emp_var_rate,	cons_price_idx,	cons_conf_idx,euribor3m,nr_employed	
                    ]])
# euribor3m,	nr_employed,
# Use the trained model to predict
    prediction = model.predict(prediction)  

    if prediction == 0:
        st.markdown('The :blue[Fraud risk] of this person is :green[Low]')
    else:
        st.markdown('The :blue[Fraud risk] of this person is :red[High]')