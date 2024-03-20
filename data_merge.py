import pandas as pd
import numpy as np
import json
import os

class FeatureProcessor():
    def __init__(self):
        pass 

    def create_main_features(self, df):
        '''input: df opened from train.csv or test.csv.
           return: processed df with necessary features generated. 
        '''
        pass      

    def create_electricity_features(self, df):
        pass 

    def create_gas_features(self, df):
        pass 

    def create_client_features(self, df):
        pass 

    def create_historical_weather(self, df):
        pass 

    def create_forecast_weather(self, df):
        pass 
    
    def __call__(self, main_data, electricity_prices, gas_prices, client, historical_weather, forecast_weather):
        '''input: a series of dataframe data. '''
        '''output: a giant DF with the input dataframe processed and merged.'''
        pass 

def main():
    # open CSV in df  
    DATA_DIR = "E:\\UCSD graduate study\\ECE225A Probability & Statistics\\projects\\predict-energy-behavior-of-prosumers"
    train= pd.read_csv(DATA_DIR + "\\train.csv")
    electricity_prices = pd.read_csv(DATA_DIR + "\\electricity_prices.csv")
    gas_prices = pd.read_csv(DATA_DIR + "\\gas_prices.csv") 
    client = pd.read_csv(DATA_DIR + "\\client.csv")
    historical_weather = pd.read_csv(DATA_DIR + "\\historical_weather.csv")
    forecast_weather = pd.read_csv(DATA_DIR + "\\forecast_weather.csv")
    

    # create the feature processor
    feature_processor = FeatureProcessor() 
    # merged_df = feature_processor(main_data, electricity_prices, gas_prices, client, historical_weather, forecast_weather)

    # the return df from the feature processor should contains:
    #   all necessary columns used for data exploration and visulization; 
    #   all necessary features used for prediction task; 
    #   contains no invalid entries; 

if __name__ == '__main__':
    main()