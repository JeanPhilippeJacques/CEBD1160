import numpy as np   # import numpy
import pandas as pd  # import pandas
import preprocessing
import feature_engineering
import warnings
import LGBMtrain
import pickle
import utils   
warnings.simplefilter(action = 'ignore', category = FutureWarning)
print('Welcome to Training Home Credit Risk')

x=input("Enter where you want to start 1:Preprocessing 2:feature Engeniring 3:LGBM Training ")
BO=input("Do you want to Use Bayesian Optimization 1:YES 0:NO ")

if x=='1':
    print('Preprocessing is stating....')
    df=preprocessing.load_data()
    
    
    df= feature_engineering.FeatEngin(df)
    print('LGMB training is stating....')
    LGBMtrain.Train(df,BO)
    
if x=='2':
    with open('train//ctdict1.pkl', 'rb') as handle:
        cvsdict = pickle.load(handle)
    print('loading preprocess data is stating....')    
    df= pd.read_csv('train//df_application_train_new1.csv',dtype=cvsdict)
    for c in df:
        print(c)
    print(df.shape)
    print('feature engineering is stating....')
    df= feature_engineering.FeatEngin(df)
    df=utils.clean(df)
    print('LGMB training is stating....')
    LGBMtrain.Train(df,BO)

if x=='3':
    with open('train//ctdict2.pkl', 'rb') as handle:
        cvsdict = pickle.load(handle)
    print('loading preprocess data is stating....')    
    df= pd.read_csv('train//df_application_train_new2.csv',dtype=cvsdict)
    for c in df:
        print(c)
    print(df.shape)
    df=utils.clean(df)
    print('LGMB training is stating....')
    LGBMtrain.Train(df,BO)