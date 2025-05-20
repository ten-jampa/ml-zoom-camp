############################### 
# In this file, I want to store the helper functions that I 
# have defined and use over the regression lab
###################################

#### Imports ####

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pylot as plt
import seaborn

## Data Preparation ##

def clean_names(some_df):
    assert isinstance(some_df, DataFrame)
    some_df = some_df.columns.str.lower().str.replace(' ', '_')
    return some_df

def clean_data_entries(some_df):
    assert isinstance(some_df, DataFrame)
    for feature in some_df.columns:
        if some_df[feature].dtypes == 'object':
            some_df[feature] = some_df[feature].str.lower().replace(' ', '_')
    return some_df


def shuffle_dataset(df, seed = 42):
    np.random.seed(42)
    indices = np.arange(len(df))
    np.random.shuffle(indices)

    #we want a split of 60/20/20
    ntrain = int(0.6 * len(df))
    nval = int(0.2 * len(df))
    ntest = len(df) - ntrain - nval
    print(f'The split of data is {ntrain, nval, ntest} with total of {len(df)} samples in the dataset')

    ## For the data matrix
    train = df.iloc[indices[:ntrain]]
    val = df.iloc[indices[ntrain: ntrain + nval]]
    test = df.iloc[indices[ntrain+nval:]]

    ## Reset the index here
    train = train.reset_index(drop = True)
    val = val.reset_index(drop = True)
    test = val.reset_index(drop = True)
    
    return train, val, test

def get_target_vector(df, feature, remove = True):
    #returns the target values as np array
    #removes the target from the data if remove is True
    ytarget = df[feature]
    if remove:
        del df[feature]
    return ytarget.values

def prepare_X(some_df, fill = 0):
    some_df = some_df.copy()
    #we are filling the null values with zero
    some_df = some_df.fillna(fill)
    X = some_df.values #np array

    ones  = np.ones(X.shape[0]) 
    X = np.column_stack([ones, X])

    return X


######### Training ###########

def train_linear_model(X, y):
    #we use normal equation
    assert X.shape[0] == y.shape[0]
    XTX = X.T.dot(X)    
    XTX_inv = np.linalg.inv(XTX)
    return XTX_inv.dot(X.T).dot(y) # the weights vector

def train_linear_model_r(X, y, r = 0.01):
    X = X + np.ones(X.shape)* r
    return train_linear_model(X, y)




############ Evaluations #############


def evaluate(Xtest, ytest, w, metric = rmse):
    yhats = linear_model(Xtest, w)
    return metric(ytest, yhats)




