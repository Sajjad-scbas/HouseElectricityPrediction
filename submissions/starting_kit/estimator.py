import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression



def categorical_col_process(df):
    """
    A sample function to ransform categorial 
    features ['WeekStatus', 'Day_of_week', 'DayType', 'InstantF'] 
    to numeric features
    """ 
    df_changed = df.copy()
    col = ['WeekStatus', 'Day_of_week', 'DayType', 'InstantF']
    for a in col:
        if a not in df:
            return False 
    
    # WeekStatus column
    df_changed['WeekStatus_numerical'] = [0 if s == 'Weekend' else 1 for s in df['WeekStatus']]
    
    # Day_of_the_week column 
    day_in_radians = 2*np.pi*df['Day_of_week'].map({'Monday': 0, 'Tuesday': 1/7, 'Wednesday': 2/7, 'Thursday': 3/7, 'Friday': 4/7, 'Saturday': 5/7, 'Sunday': 6/7})
    df_changed['day_of_week_sin'] = np.sin(day_in_radians)
    df_changed['day_of_week_cos'] = np.cos(day_in_radians)

    df_changed.drop(columns=col, inplace=True)
    return df_changed

def mean_filling_values(df):
    """
    A function to fill na values present in RH_6 and Visibility 
    features with mean values
    """
    df_imputed = df.copy()
    # Fill in the missing values with the mean of each column
    df_imputed['RH_6'].fillna(df_imputed['RH_6'].mean(), inplace=True)
    df_imputed['Visibility'].fillna(df_imputed['Visibility'].mean(), inplace=True)
    return df_imputed


class FeatureExtractor(BaseEstimator):
    """
    A class to put all your costum manipulation on data
    """
    def fit(self, X, y):
        return self

    def transform(self, X):
        X = categorical_col_process(X) 
        X = mean_filling_values(X)
        return 



def get_estimator():

    '''We define all transformation on the data'''
    feature_extractor = FeatureExtractor()

    '''We define a Regressor'''
    regressor = LinearRegression() 

    '''We wrap all in a pipeline'''
    pipe = make_pipeline(feature_extractor, StandardScaler(), regressor)
    return pipe
