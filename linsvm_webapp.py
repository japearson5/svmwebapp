# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:45:25 2022

@author: JPear
"""

import pandas as pd
import numpy as np
import pickle
import streamlit as st

from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

#importing the housing data
housingdata = pd.read_csv("housing-data.csv")
df = pd.DataFrame(housingdata)
df['total_bedrooms']=df.groupby('ocean_proximity')['total_bedrooms'].apply(lambda x: x.fillna(x.median()))

#using z-score to get rid of outlier data
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].apply(zscore)

#getting rid of values with z score greater than 3, removing extreme outliers
outliers = np.where(df[numeric_cols].apply(zscore) >= 3)

#print(outliers)
df.drop(outliers[0], inplace = True)

#adding column to show average number of rooms per house
df.insert(3, 'rooms_per_house', df.values[:,3]/df.values[:,6], True)
df.insert(4, 'bedrooms_per_house', df.values[:,5]/df.values[:,7], True)
df['rooms_per_house'] = df['rooms_per_house'].apply(pd.to_numeric, errors='coerce')
df['bedrooms_per_house'] = df['bedrooms_per_house'].apply(pd.to_numeric, errors='coerce')

#mapping object column of ocean_proximity to numbers
df.ocean_proximity = df.ocean_proximity.map({'ISLAND':0, 'NEAR OCEAN':1, 'NEAR BAY':2, '<1H OCEAN':3, 'INLAND':4})

#doing feature selection, deciding which variables may not be useful for our model
del df['total_rooms']
del df['total_bedrooms']

#binning the median house values into 4 different categories
bin_labels = [0, 1, 2, 3] #'<100k', '100k-250k', '250k-400k', '>400k'
cut_bins = [0, 100000, 250000, 400000, 600000]
df['median_house_value'] = pd.cut(df['median_house_value'], bins = cut_bins, labels = bin_labels)
X = df.drop(columns = 'median_house_value', axis = 1)

#standardizing the data to a similar range so the ML model can more quickly and accurately assess the data
scaler = StandardScaler()
scaler.fit(X)


#loading the model
loaded_model = pickle.load(open('linsvm.sav', 'rb'))

#creating a function for the prediction
def housing_price_prediction(input_data):
    #making a predictive system
    #input_data = (-122.23, 37.88, 41, 6.98412698413, 1.02380952381, 322, 126, 8.3252, 2)
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array for this one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(std_data) #std_data
    #print(prediction)

    if (prediction[0] == 0):
        return 'House is less than 100,000 dollars.'
    elif (prediction[0] == 1):
        return 'House is between 100,000 and 250,000 dollars.'
    elif (prediction[0] == 2):
        return 'House is between 250,000 and 400,000 dollars.'
    else:
        return 'House is more than 400,000 dollars.'
    
def main():
    #giving a title
    st.title('California Housing Price Predictor')
    #st.image('C:/Users/JPear/cs551/streamlit/calineighborhood.jpg', width = 890)
    
    #get data from user
    longitude = st.slider('Choose the longitude:', -124.35, -114.31)#, value=None, step=None, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible")
    latitude = st.slider('Choose the latitude:', 32.54, 41.95)
    housing_median_age = st.slider('Housing median age on the block:', 1, 52)
    rooms_per_house = st.number_input('Average number of rooms per house on the block:', 0.5, 142.0)
    bedrooms_per_house = st.number_input('Average number of bedrooms per house on the block:', 0.4, 35.0)
    population = st.number_input('Population of the block:', 3, 4820)
    households = st.number_input('Number of houses on the block:', 2, 1645)
    median_income = st.number_input('Median income of homeowners on the block (in tens of thousands of dollars per year):', 0.48, 9.6)
    ocean_proximity = st.slider('Proximity to the ocean (0 = On an Island, 1 = Near the Ocean, 2 = Near the Bay, 3 = Less than an hour drive, 4 = Inland):', 0, 4)
    
    result = ''
    
    #the prediction button
    if st.button('Find Approximate Housing Value'):
        result = housing_price_prediction([longitude, latitude, housing_median_age, rooms_per_house, bedrooms_per_house, population, households, median_income, ocean_proximity])
    
    st.success(result)
    
if __name__ == '__main__':
    main()