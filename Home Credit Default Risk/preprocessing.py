"""
all functions related to preprocessing
"""
import os
import pandas as pd
import gc
import glob
import utils
import config
import numpy as np
import utils
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

# lebal encoding dumm
from sklearn.preprocessing import LabelEncoder

# Super load file
# We want to walk through the dir and get all file names once
DATA_PATH = '' #config.DATA_PATH

def load_data():
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    file_names = glob.glob( r"*.csv")
    print('Will load and aggregate')
    print(file_names)

    myDf = []
    df_application_train=[]
    for f in file_names:
        fstr = f.replace(DATA_PATH, '')
        fstr = fstr.replace('.csv', '')
        globals()['df_' + fstr] = pd.read_csv(f)
        print('df_' + fstr)
        mem_use = globals()['df_' + fstr].memory_usage().sum() / 1024 ** 3
        print('Memory usage of dataframe is {:.6f} GB'.format(mem_use))
        utils.reducemem(globals()['df_' + fstr])
        mem_use2 = globals()['df_' + fstr].memory_usage().sum() / 1024 ** 3
        print('Memory usage of dataframe after is {:.6f} GB'.format(mem_use2))
        print('We save {:.6f} GB'.format(mem_use - mem_use2))
        myDf.append('df_' + fstr)
        # print(globals()['df_' + fstr].head(1))
        # print(globals()['df_' + fstr].info())
    gc.collect()


    # oder df to proceses data cleaning and agg loop
    
    
    # list and dictionary for cleaning agg and merge process
    TsDf = ['df_bureau_balance', 'df_credit_card_balance', 'df_POS_CASH_balance']
    KEY = {'df_bureau': 'SK_ID_CURR', 'df_bureau_balance': 'SK_ID_BUREAU', 'df_credit_card_balance': 'SK_ID_PREV',
           'df_application_train': '',
           'df_installments_payments': 'SK_ID_PREV', 'df_POS_CASH_balance': 'SK_ID_PREV',
           'df_previous_application': 'SK_ID_CURR'}
    Merge = {'df_bureau_balance': 'df_bureau', 'df_POS_CASH_balance': 'df_previous_application',
             'df_installments_payments': 'df_previous_application', 'df_credit_card_balance': 'df_previous_application'}
    
    gc.collect()
    
    # clean 'df_bureau' to 'df_previous_application'
    # agg df_credit_card_balance df_installments_payments df_POS_CASH_balance and merge with df_previous_application
    # agg df_bureau_balance and merge with df_bureau
    for df in myDf:
        print(df)
        yourdf = globals()[df].copy()
        # rename column with name of df
        for c in yourdf:
            if c != 'SK_ID_CURR' and c != 'SK_ID_PREV' and c != 'TARGET' and c != 'SK_ID_BUREAU':
                yourdf = yourdf.rename(columns={c: df + "_" + c})
    
        # Identifying all categorical features
        features = [f for f in yourdf.columns.values if f not in ['TARGET']]
        cat_features = []
    
        for f in features:
            if str(yourdf[f].dtype) in ['int16', 'int32', 'int64']:
                cat_features.append(f)
    
        non_categ = [""]
    
        cat_features = list(set(cat_features) - set(non_categ))
    # no more object type column/didn't use get dummy yet
    
        # Fillna object colect num col
        objectcolumns = []
        for f in yourdf:
            if str(yourdf[f].dtype) == 'object':
                objectcolumns.append(f)
        yourdf[objectcolumns] = yourdf[objectcolumns].replace(np.nan, 'UNKNOW', regex=True)
        print(objectcolumns)
        # Fillna num colect num col
    
        num_feat = []
        for f in yourdf:
            if f not in objectcolumns:
                num_feat.append(f)
    
        for f in num_feat:
            yourdf[f].fillna(yourdf[f].median(), inplace=True)
    
    
        labelencoders = {}
    
        for name in objectcolumns:
            if yourdf[name].nunique() < 20:
                dumm = pd.get_dummies(yourdf[name], prefix=name)
                yourdf = pd.concat([yourdf, dumm], axis=1)
                # print(name)
                # print(dumm.head())
    
            yourdf[name] = yourdf[name].astype('str')
            labelencoder = LabelEncoder()
            labelencoder.fit(yourdf[[name]])
            yourdf[name] = labelencoder.transform(yourdf[[name]])
            labelencoders[name] = labelencoder
        
        # create dictionnary for group agg
        agg_dict = {}
        for i in yourdf.columns:
            if i in objectcolumns:
                agg_dict[i] = ['max','min','nunique']
            else:
                if i != 'SK_ID_CURR' and i != 'SK_ID_PREV' and i != 'SK_ID_BUREAU':
                    agg_dict[i] = ['max', 'sum', 'mean']
    
        # aggregation of the df_bureau_balance','df_credit_card_balance','df_POS_CASH_balance , df_installments_payments
    
        if KEY[df] == "SK_ID_PREV" or KEY[df] == "SK_ID_BUREAU":
            if df in TsDf:
                yourdfgrp = yourdf.groupby(KEY[df], as_index=False).agg({**agg_dict})
                yourdfgrp.columns = ['{}{}'.format(x[0], x[1]) for x in yourdfgrp.columns.tolist()]
                yourdfgrp = yourdfgrp.reset_index(drop=True)
                # yourdfgrp= yourdfgrp.rename(columns={"SK_ID_CURR":'SK_ID_CURR2'})
    
                yourdf1 = yourdf[yourdf[df + '_MONTHS_BALANCE'] == -1]
                yourdf1grp = yourdf1.groupby(KEY[df], as_index=False).agg({**agg_dict})
                yourdf1grp.columns = ['{}{}'.format(x[0], x[1]) for x in yourdf1grp.columns.tolist()]
                yourdf1grp = yourdf1grp.reset_index(drop=True)
                for c in yourdf1grp:
                    if c != KEY[df]:
                        yourdf1grp = yourdf1grp.rename(columns={c: "Last_" + c})
    
                yourdf6 = yourdf[yourdf[df + '_MONTHS_BALANCE'] >= -6]
                yourdf6grp = yourdf6.groupby(KEY[df], as_index=False).agg({**agg_dict})
                yourdf6grp.columns = ['{}{}'.format(x[0], x[1]) for x in yourdf6grp.columns.tolist()]
                yourdf6grp = yourdf6grp.reset_index(drop=True)
                for c in yourdf6grp:
                    if c != KEY[df]:
                        yourdf6grp = yourdf6grp.rename(columns={c: "Last_6_" + c})
    
                yourdf12 = yourdf[yourdf[df + '_MONTHS_BALANCE'] == -12]
                yourdf12grp = yourdf12.groupby(KEY[df], as_index=False).agg({**agg_dict})
                yourdf12grp.columns = ['{}{}'.format(x[0], x[1]) for x in yourdf12grp.columns.tolist()]
                yourdf12grp = yourdf12grp.reset_index(drop=True)
                for c in yourdf12grp:
                    if c != KEY[df]:
                        yourdf12grp = yourdf12grp.rename(columns={c: "Last_12M_Diff" + c})
                        yourdf12grp["Last_12M_Diff" + c] = yourdf1grp["Last_" + c] - yourdf12grp[
                            "Last_12M_Diff" + c]
    
                yourdfgrp = pd.merge(yourdfgrp, yourdf1grp, on=KEY[df], how='left')
                yourdfgrp = yourdfgrp.reset_index(drop=True)
                yourdfgrp = pd.merge(yourdfgrp, yourdf12grp, on=KEY[df], how='left')
                yourdfgrp = yourdfgrp.reset_index(drop=True)
                yourdfgrp = pd.merge(yourdfgrp, yourdf6grp, on=KEY[df], how='left')
                yourdfgrp = yourdfgrp.reset_index(drop=True)
                print(yourdfgrp.shape)
                print(yourdfgrp.shape)
                globals()[Merge[df]] = pd.merge(globals()[Merge[df]], yourdfgrp, on=KEY[df], how='left')
                globals()[Merge[df]] = globals()[Merge[df]].reset_index(drop=True)
                print(Merge[df])
                utils.reducemem(globals()[Merge[df]])
                gc.collect()
                del df
            else:
                yourdfgrp = yourdf.groupby(KEY[df], as_index=False).agg({**agg_dict})
                yourdfgrp.columns = ['{}{}'.format(x[0], x[1]) for x in yourdfgrp.columns.tolist()]
                yourdfgrp = yourdfgrp.reset_index(drop=True)
                # yourdfgrp= yourdfgrp.rename(columns={"SK_ID_CURR":df+'SK_ID_CURR'})
                globals()[Merge[df]] = pd.merge(globals()[Merge[df]], yourdfgrp, on='SK_ID_PREV', how='left')
                globals()[Merge[df]] = globals()[Merge[df]].reset_index(drop=True)
                print(Merge[df])
                utils.reducemem(globals()[Merge[df]])
                gc.collect()
            
        else:
            globals()[df] = yourdf.copy()
            globals()[df] = globals()[df].reset_index(drop=True)
            utils.reducemem(globals()[df])
            gc.collect()
            
    yourdf=[]
    dumm=[]
    features=[]
    cat_features=[]
    objectcolumns=[]
    agg_dict=[]
    yourdfgrp=[]
    yourdf1=[]
    yourdf1grp=[]
    yourdf6=[]
    yourdf6grp=[]
    yourdf12=[]
    yourdf12grp=[]
    gc.collect()
    del globals()['df_bureau_balance'] , globals()['df_credit_card_balance'] ,globals()['df_installments_payments']   , globals()['df_POS_CASH_balance'] 
    gc.collect() 
     
    
    # agg df_bureau and df_previous_application and merge with df_application_train
    for df in myDf[1:]:
        if KEY[df] == "SK_ID_CURR":
            yourdf = globals()[df].copy()
            # create dictionnary for group agg
            agg_dict = {}
            for i in yourdf.columns:
                if i != 'SK_ID_CURR' and i != 'SK_ID_PREV':
                    agg_dict[i] = ['max', 'sum', 'nunique']
    
            yourdf = yourdf.reset_index(drop=True)
            yourdfgrp = yourdf.groupby('SK_ID_CURR', as_index=False).agg({**agg_dict})
            yourdfgrp.columns = ['{}{}'.format(x[0], x[1]) for x in yourdfgrp.columns.tolist()]
            yourdfgrp = yourdfgrp.reset_index(drop=True)
            print(yourdfgrp.shape)
            globals()['df_application_train'] = pd.merge(globals()['df_application_train'], yourdfgrp, on='SK_ID_CURR', how='left')
            print(df)
            print(globals()['df_application_train'].shape)
            gc.collect()
            del df
            utils.reducemem(globals()['df_application_train'])
    globals()['df_application_train'] = globals()['df_application_train'].replace(-np.Inf, 0)
    globals()['df_application_train'] = globals()['df_application_train'].replace(np.Inf, 0)
    globals()['df_application_train'] = globals()['df_application_train'].replace(np.nan, 0)

   
    yourdf=[]
    dumm=[]=[]
    features=[]
    cat_features=[]
    objectcolumns=[]
    agg_dict=[]
    yourdfgrp=[]
    yourdf1=[]
    yourdf1grp=[]
    yourdf6=[]
    yourdf6grp=[]
    yourdf12=[]
    yourdf12grp=[]
    gc.collect()
    utils.reducemem(globals()['df_application_train'])
    gc.collect()
   
    globals()['df_application_train'].to_csv('train//df_application_train_new1.csv',index=False)
    CTdict = {}
    for i in globals()['df_application_train'].columns: 
        CTdict[i] = globals()['df_application_train'][i].dtypes
        
    import pickle
    f = open("train//ctdict1.pkl","wb")
    pickle.dump(CTdict,f)
    f.close()
    return globals()['df_application_train']
