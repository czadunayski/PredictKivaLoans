# -*- coding: utf-8 -*-
"""
November2022
caileyzadunayski

"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

st.header("Kiva Loans Prediction App")
st.text_input("Enter your Name: ", key="name")
kiva_loans = pd.read_csv("https://raw.githubusercontent.com/kristophernerl/PredictKivaLoans/main/kiva_loans2.csv")

#load label encoders
label_encoder_sector = LabelEncoder()
label_encoder_sector.classes_ = np.load('label_encoder_sector.npy',allow_pickle=True)
label_encoder_gender = LabelEncoder()
label_encoder_gender.classes_ = np.load('label_encoder_gender.npy',allow_pickle=True)

#load model from pickle file
import joblib
model = joblib.load("rfmodel.pkl")

#Allowing users to see the training dataframe if desired
if st.checkbox('Show Training Dataframe'):
    kiva_loans

##start
st.subheader("Please select the appropriate loan details to receive a predicated loan amount")
left_column, right_column = st.columns(2)
with left_column:
    inpu_sector = st.radio(
        'What will your loan be used for?',
        np.unique(kiva_loans['sector']))

with right_column:
    inpu_gender = st.radio(
        'What is/are the gender of the recipient(s)? (F) female, (M) male, or (B) both',
        np.unique(kiva_loans['gender_bin']))

#Creating Input Country Dropdown
country = (
        'Armenia',
        'Bolivia',
        'Cambodia',
        'Cameroon',
        'Colombia',
        'Ecuador',
        'Egypt',
        'El Salvador',
        'Ghana',
        'Guatemala',
        'Haiti',
        'Honduras',
        'India',
        'Indonesia',
        'Jordan',
        'Kenya',
        'Kyrgyzstan',
        'Lebanon',
        'Liberia',
        'Madagascar',
        'Mali',
        'Mexico',
        'Mozambique',
        'Nicaragua',
        'Nigeria',
        'Pakistan',
        'Paraguay',
        'Peru',
        'Philippines',
        'Samoa',
        'Sierra Leone',
        'Tajikistan',
        'Tanzania',
        'Timor-Leste',
        'Togo',
        'Turkey',
        'Uganda',
        'Vietnam',
        'Zimbabwe'
        )

country = st.selectbox("Select your Country", country)

#coding inputted Country to Income Index
index=500
rate=1
currency='USD'

if country == 'Armenia':
  index = 0.681
  rate = 394.59
  currency = 'AMD'
elif country == 'Bolivia':
    index = 0.634
    rate = 6.89
    currency = 'BOB'
elif country == 'Cambodia':
    index = 0.533
    rate = 4127.68
    currency = 'KHR'
elif country == 'Cameroon':
    index = 0.526
    rate = 635.3
    currency = 'XAF'
elif country == 'Colombia':
    index = 0.735
    rate = 4785.86
    currency = 'COP'
elif country == 'Ecuador':
    index = 0.699
    rate = 26545.77
    currency = 'ECS'
elif country == 'Egypt':
    index = 0.703
    rate = 24.3
    currency = 'EGP'
elif country == 'El Salvador':
    index = 0.638
    rate = 8.7474
    currency = 'SVC'
elif country == 'Ghana':
    index = 0.555
    rate = 14.45
    currency = 'GHS'
elif country == 'Guatemala':
    index = 0.648
    rate = 7.79
    currency = 'GTQ'
elif country == 'Haiti':
    index = 0.425
    rate = 135.03
    currency = 'HTG'
elif country == 'Honduras':
    index = 0.563
    rate = 24.63
    currency = 'HNL'
elif country == 'India':
    index = 0.629
    rate = 80.59
    currency = 'INR'
elif country == 'Indonesia':
    index = 0.707
    rate = 15506.35
    currency = 'IDR'
elif country == 'Jordan':
    index = 0.667
    rate = 0.71
    currency = 'JOD'
elif country == 'Kenya':
    index = 0.511
    rate = 121.48
    currency = 'KES'
elif country == 'Kyrgyzstan':
    index = 0.523
    rate = 84.3
    currency = 'KGS'
elif country == 'Lebanon':
    index = 0.719
    rate = 1507.06
    currency = 'LBP'
elif country == 'Liberia':
    index = 0.36
    rate = 153.8
    currency = 'LRD'
elif country == 'Madagascar':
    index = 0.396
    rate = 4275.44
    currency = 'MGA'
elif country == 'Mali':
    index = 0.45
    rate = 633.07
    currency = 'XOF'
elif country == 'Mexico':
    index = 0.78
    rate = 19.54
    currency = 'MXN'
elif country == 'Mozambique':
    index = 0.367
    rate = 63.83
    currency = 'MZN'
elif country == 'Nicaragua':
    index = 0.592
    rate = 35.87
    currency = 'NIO'
elif country == 'Nigeria':
    index = 0.597
    rate = 440.96
    currency = 'NGN'
elif country == 'Pakistan':
    index = 0.592
    rate = 220.75
    currency = 220.75
elif country == 'Paraguay':
    index = 0.716
    rate = 7110.14
    currency = 'PYG'
elif country == 'Peru':
    index = 0.723
    rate = 3.84
    currency = 'PEN'
elif country == 'Philippines':
    index = 0.682
    rate = 57.36
    currency = 'PHP'
elif country == 'Samoa':
    index = 0.615
    rate = 2.7939
    currency = 'Samoan Tala'
elif country == 'Sierra Leone':
    index = 0.394
    rate = 17820
    currency = 'SLL'
elif country == 'Tajikistan':
    index = 0.526
    rate = 10.01
    currency = 'TJS'
elif country =='Tanzania':
    index = 0.5
    rate = 2326.16
    currency = 'TZS'
elif country == 'Timor-Leste':
    index = 0.651
    rate = 1
    currency = 'USD - timor leste does not have their own banknote'
elif country == 'Togo':
    index = 0.415
    rate = 633.07
    currency = 'XOF'
elif country == 'Turkey':
    index = 0.832
    rate = 18.57
    currency = 'TRY'
elif country == 'Uganda':
    index = 0.43
    rate = 3751.84
    currency = 'UGX'
elif country == 'Vietnam':
    index = 0.616
    rate = 24780
    currency = 'VND'
elif country == 'Zimbabwe':
    index = 0.475
    rate = 322
    currency = 'ZWL'

#creating Input Monthly Payment Slider
monthpay = st.slider('How much can you afford per month (in your local currency)', 1.25, (rate*max(kiva_loans["repayment_per_mo"])), 1.0)

#creating Prediction (button)

input_payment = monthpay/rate
input_income = index

if st.button('Make Prediction'):
    input_sector = label_encoder_sector.transform(np.expand_dims(inpu_sector, -1))
    input_gender = label_encoder_gender.transform(np.expand_dims(inpu_gender, -1))
    inputs = np.expand_dims(
        [int(input_sector), int(input_gender), input_payment, input_income], 0)
    prediction = model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"Your recomended loan amount to request is: {rate*(np.squeeze(prediction, -1)):.2f} {currency} or ${(np.squeeze(prediction, -1)):.2f}USD")

    st.write(f"Thank you {st.session_state.name}! Please run again if you're curious to see the results for other loan details")
    

st.bar_chart(kiva_loans)
