import pandas as pd
import numpy as np
import gc
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import pickle
import utils
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import config
"""
all functions about feature engineering or analysis
"""

def FeatEngin(df):   
   
    #Features

    df['df_application_train_AMT_CREDIT * df_application_train_AMT_ANNUITY '] = df['df_application_train_AMT_CREDIT'] * df['df_application_train_AMT_ANNUITY']
    df['df_application_train_AMT_INCOME_TOTAL * df_application_train_AMT_ANNUITY'] = df['df_application_train_AMT_INCOME_TOTAL'] * df['df_application_train_AMT_ANNUITY']
    df['df_application_train_AMT_INCOME_TOTAL * df_application_train_DAYS_EMPLOYED'] = df['df_application_train_AMT_INCOME_TOTAL'] * df['df_application_train_DAYS_EMPLOYED']
    df['df_application_train_AMT_CREDIT * df_application_train_DAYS_EMPLOYED'] = df['df_application_train_AMT_CREDIT'] * df['df_application_train_DAYS_EMPLOYED']
    df['df_application_train_AMT_ANNUITY * df_application_train_DAYS_EMPLOYED'] = df['df_application_train_AMT_ANNUITY'] * df['df_application_train_DAYS_EMPLOYED']
    df['app missing'] = df.isnull().sum(axis = 1).values
    df['app EXT_SOURCE mean'] = df[['df_application_train_EXT_SOURCE_1', 'df_application_train_EXT_SOURCE_2',
                                    'df_application_train_EXT_SOURCE_3']].mean(axis = 1)
    df['app EXT_SOURCE std'] = df[['df_application_train_EXT_SOURCE_1', 'df_application_train_EXT_SOURCE_2',
                                   'df_application_train_EXT_SOURCE_3']].std(axis = 1)
    df['app EXT_SOURCE prod'] = df['df_application_train_EXT_SOURCE_1'] * df['df_application_train_EXT_SOURCE_2'] * df['df_application_train_EXT_SOURCE_3']
    df['app EXT_SOURCE_1xEXT_SOURCE_2'] = df['df_application_train_EXT_SOURCE_1'] * df['df_application_train_EXT_SOURCE_2']
    df['app EXT_SOURCE_1xEXT_SOURCE_3'] = df['df_application_train_EXT_SOURCE_1'] * df['df_application_train_EXT_SOURCE_3']
    df['app EXT_SOURCE_2xEXT_SOURCE_3'] = df['df_application_train_EXT_SOURCE_2'] * df['df_application_train_EXT_SOURCE_3']
    df['app EXT_SOURCE_1xDAYS_EMPLOYED'] = df['df_application_train_EXT_SOURCE_1'] * df['df_application_train_DAYS_EMPLOYED']
    df['app EXT_SOURCE_2xDAYS_EMPLOYED'] = df['df_application_train_EXT_SOURCE_2'] * df['df_application_train_DAYS_EMPLOYED']
    df['app EXT_SOURCE_3xDAYS_EMPLOYED'] = df['df_application_train_EXT_SOURCE_3'] * df['df_application_train_DAYS_EMPLOYED']
    df['app EXT_SOURCE_1_div_DAYS_BIRTH'] = df['df_application_train_EXT_SOURCE_1'] / df['df_application_train_DAYS_BIRTH']
    df['app EXT_SOURCE_2_div_DAYS_BIRTH'] = df['df_application_train_EXT_SOURCE_2'] / df['df_application_train_DAYS_BIRTH']
    df['app EXT_SOURCE_3_div_DAYS_BIRTH'] = df['df_application_train_EXT_SOURCE_3'] / df['df_application_train_DAYS_BIRTH']
    df['app AMT_CREDITminusT_GOODS_PRICE'] = df['df_application_train_AMT_CREDIT'] - df['df_application_train_AMT_GOODS_PRICE']
    df['app AMT_CREDIT_div_AMT_GOODS_PRICE'] = df['df_application_train_AMT_CREDIT'] / df['df_application_train_AMT_GOODS_PRICE']
    df['app AMT_CREDIT_div_AMT_ANNUITY'] = df['df_application_train_AMT_CREDIT'] / df['df_application_train_AMT_ANNUITY']
    df['app AMT_CREDIT_div_AMT_INCOME_TOTAL'] = df['df_application_train_AMT_CREDIT'] / df['df_application_train_AMT_INCOME_TOTAL']
    df['df_application_train_AMT_INCOME_TOTAL / 12 - df_application_train_AMT_ANNUITY'] = df['df_application_train_AMT_INCOME_TOTAL'] / 12. - df['df_application_train_AMT_ANNUITY']
    df['df_application_train_AMT_INCOME_TOTAL / df_application_train_AMT_ANNUITY'] = df['df_application_train_AMT_INCOME_TOTAL'] / df['df_application_train_AMT_ANNUITY']
    df['df_application_train_AMT_INCOME_TOTAL - df_application_train_AMT_GOODS_PRICE'] = df['df_application_train_AMT_INCOME_TOTAL'] - df['df_application_train_AMT_GOODS_PRICE']
    df['df_application_train_AMT_INCOME_TOTAL / df_application_train_CNT_CHILDREN'] = df['df_application_train_AMT_INCOME_TOTAL'] / (1 + df['df_application_train_CNT_CHILDREN'])
    df['df_application_train_AMT_CREDIT - df_application_train_AMT_GOODS_PRICE'] = df['df_application_train_AMT_CREDIT'] - df['df_application_train_AMT_GOODS_PRICE']
    df['df_application_train_AMT_CREDIT / df_application_train_AMT_GOODS_PRICE'] = df['df_application_train_AMT_CREDIT'] / df['df_application_train_AMT_GOODS_PRICE']
    df['df_application_train_AMT_CREDIT / df_application_train_AMT_ANNUITY'] = df['df_application_train_AMT_CREDIT'] / df['df_application_train_AMT_ANNUITY']
    df['df_application_train_AMT_CREDIT/ df_application_train_AMT_INCOME_TOTAL'] = df['df_application_train_AMT_CREDIT'] / df['df_application_train_AMT_INCOME_TOTAL']
    df['df_application_train_AMT_GOODS_PRICE'] = df['df_application_train_AMT_GOODS_PRICE'] \
                        .isin([225000, 450000, 675000, 900000]).map({True: 1, False: 0})
    df['df_application_train_AMT_GOODS_PRICE'] = df['df_application_train_AMT_GOODS_PRICE'] \
                        .isin([1125000, 1350000, 1575000, 1800000, 2250000]).map({True: 1, False: 0})
    df['df_application_train_OWN_CAR_AGE / df_application_train_DAYS_BIRTH'] = df['df_application_train_OWN_CAR_AGE'] / df['df_application_train_DAYS_BIRTH']
    df['df_application_train_OWN_CAR_AGE / df_application_train_DAYS_EMPLOYED'] = df['df_application_train_OWN_CAR_AGE'] / df['df_application_train_DAYS_EMPLOYED']
    df['df_application_train_OWN_CAR_AGE / df_application_train_DAYS_BIRTH'] = df['df_application_train_OWN_CAR_AGE'] / df['df_application_train_DAYS_BIRTH']
    df['df_application_train_OWN_CAR_AGE / df_application_train_DAYS_EMPLOYED'] = df['df_application_train_OWN_CAR_AGE'] / df['df_application_train_DAYS_EMPLOYED']
    df['df_application_train_DAYS_EMPLOYED - df_application_train_DAYS_BIRTH'] = df['df_application_train_DAYS_EMPLOYED'] - df['df_application_train_DAYS_BIRTH']
    df['df_application_train_DAYS_EMPLOYED / df_application_train_DAYS_BIRTH'] = df['df_application_train_DAYS_EMPLOYED'] / df['df_application_train_DAYS_BIRTH']
    df['df_application_train_CNT_CHILDREN / df_application_train_CNT_FAM_MEMBERS'] = df['df_application_train_CNT_CHILDREN'] / df['df_application_train_CNT_FAM_MEMBERS']
    df['df_bureau_AMT_CREDIT_SUMsum - df_bureau_AMT_CREDIT_SUM_DEBTsum'] = df['df_bureau_AMT_CREDIT_SUMsum'] - df['df_bureau_AMT_CREDIT_SUM_DEBTsum']
    df['df_bureau_AMT_CREDIT_SUMsum - df_bureau_AMT_CREDIT_SUM_LIMITsum'] = df['df_bureau_AMT_CREDIT_SUMsum'] - df['df_bureau_AMT_CREDIT_SUM_LIMITsum']
    df['df_bureau_AMT_CREDIT_SUMsum - df_bureau_AMT_CREDIT_SUM_OVERDUEsum'] = df['df_bureau_AMT_CREDIT_SUMsum'] - df['df_bureau_AMT_CREDIT_SUM_OVERDUEsum']
    df['df_bureau_DAYS_CREDITsum - df_bureau_CREDIT_DAY_OVERDUEsum'] = df['df_bureau_DAYS_CREDITsum'] - df['df_bureau_CREDIT_DAY_OVERDUEsum']
    df['df_bureau_DAYS_CREDITsum - df_bureau_DAYS_CREDIT_ENDDATEsum'] = df['df_bureau_DAYS_CREDITsum'] - df['df_bureau_DAYS_CREDIT_ENDDATEsum']
    df['df_bureau_DAYS_CREDITsum - df_bureau_DAYS_ENDDATE_FACTsum'] = df['df_bureau_DAYS_CREDITsum'] - df['df_bureau_DAYS_ENDDATE_FACTsum']
    df['df_bureau_DAYS_CREDIT_ENDDATEsum - df_bureau_DAYS_ENDDATE_FACTsum'] = df['df_bureau_DAYS_CREDIT_ENDDATEsum'] - df['df_bureau_DAYS_ENDDATE_FACTsum']
    df['df_bureau_DAYS_CREDIT_UPDATEsum - df_bureau_DAYS_CREDIT_ENDDATEsum'] = df['df_bureau_DAYS_CREDIT_UPDATEsum'] - df['df_bureau_DAYS_CREDIT_ENDDATEsum']
    df['df_previous_application_AMT_APPLICATIONsum / df_previous_application_AMT_CREDITsum'] = df['df_previous_application_AMT_APPLICATIONsum'] / df['df_previous_application_AMT_CREDITsum']
    df['df_previous_application_AMT_APPLICATIONsum - df_previous_application_AMT_CREDITsum'] = df['df_previous_application_AMT_APPLICATIONsum'] - df['df_previous_application_AMT_CREDITsum']
    df['df_previous_application_AMT_APPLICATIONsum - df_previous_application_AMT_GOODS_PRICEsum'] = df['df_previous_application_AMT_APPLICATIONsum'] - df['df_previous_application_AMT_GOODS_PRICEsum']
    df['df_previous_application_AMT_GOODS_PRICEsum - df_previous_application_AMT_CREDITsum'] = df['df_previous_application_AMT_GOODS_PRICEsum'] - df['df_previous_application_AMT_CREDITsum']
    df['df_previous_application_DAYS_FIRST_DRAWINGsum - df_previous_application_DAYS_FIRST_DUEsum'] = df['df_previous_application_DAYS_FIRST_DRAWINGsum'] - df['df_previous_application_DAYS_FIRST_DUEsum']
    df['df_previous_application_DAYS_TERMINATIONsum less -500'] = (df['df_previous_application_DAYS_TERMINATIONsum'] < -500).astype(int)
    df['df_previous_application_df_credit_card_balance_AMT_TOTAL_RECEIVABLEsumsum - df_previous_application_df_credit_card_balance_AMT_RECIVABLEsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_TOTAL_RECEIVABLEsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_RECIVABLEsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_TOTAL_RECEIVABLEsumsum - df_previous_application_df_credit_card_balance_AMT_RECEIVABLE_PRINCIPALsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_TOTAL_RECEIVABLEsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_RECEIVABLE_PRINCIPALsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_RECIVABLEsumsum - df_previous_application_df_credit_card_balance_AMT_RECEIVABLE_PRINCIPALsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_RECIVABLEsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_RECEIVABLE_PRINCIPALsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_BALANCEsumsum - df_previous_application_df_credit_card_balance_AMT_RECIVABLEsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_BALANCEsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_RECIVABLEsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_BALANCEsumsum - df_previous_application_df_credit_card_balance_AMT_RECEIVABLE_PRINCIPALsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_BALANCEsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_RECEIVABLE_PRINCIPALsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_BALANCEsumsum - df_previous_application_df_credit_card_balance_AMT_TOTAL_RECEIVABLEsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_BALANCEsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_TOTAL_RECEIVABLEsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_CURRENTsumsum - df_previous_application_df_credit_card_balance_AMT_DRAWINGS_ATM_CURRENTsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_CURRENTsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_ATM_CURRENTsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_CURRENTsumsum - df_previous_application_df_credit_card_balance_AMT_DRAWINGS_OTHER_CURRENTsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_CURRENTsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_OTHER_CURRENTsumsum']
    df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_CURRENTsumsum - df_previous_application_df_credit_card_balance_AMT_DRAWINGS_POS_CURRENTsumsum'] = df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_CURRENTsumsum'] - df['df_previous_application_df_credit_card_balance_AMT_DRAWINGS_POS_CURRENTsumsum']
    
    def trim(st):
        st1 = st.strip(" ")
        st1 = st1.replace(',', '')
        st1 = st1.replace(' ', '_')
        st1 = st1.replace(':', '_')
        return st1
    
    
    df = df.rename(columns=lambda x: trim(x))
    df=df.replace(-np.Inf, 0)
    df=df.replace(np.Inf, 0)
    df=df.replace(np.nan, 0)
    gc.collect()
    y_train=df['TARGET']
    X_train=df.drop(columns=['TARGET'])
    del df
    bestfeat=pd.DataFrame()
    fvalue_selector = SelectKBest(f_classif, k=min(X_train.shape[1],500))
    FX_kbest = fvalue_selector.fit_transform(X_train, y_train)
    bestlist=[]
    gc.collect()
    for b in (zip(fvalue_selector.get_support(),X_train.columns)):
        if b[0]==True and b[1]!='TARGET':
            bestlist.append(b[1])
            
    var=bestlist
    
    for opp in range(1,6):   
        print(opp)
        i=0
    
        for c in var[:-1]:
            i=i+1
            Newfeat=pd.DataFrame()
            for cc in var[i:]:
                if opp ==1:
                    Newfeat[c+"_plus_"+cc]=X_train[c]+X_train[cc]
                if opp ==2:
                    Newfeat[c+"_minus_"+cc]=X_train[c]-X_train[cc]
                if opp ==3:
                    Newfeat[c+"_multi_"+cc]=X_train[c]*X_train[cc]
                if opp ==4:
                    Newfeat[c+"_div_"+cc]=X_train[c]/X_train[cc]
                if opp ==5:
                    Newfeat[c+"_div_"+cc]=X_train[c]/X_train[cc]
            if i > 1:
                Newfeat=pd.concat([Newfeat,bestfeat],axis=1)
            
            Newfeat=Newfeat.replace(-np.Inf, 0)
            Newfeat=Newfeat.replace(np.Inf, 0)
            Newfeat=Newfeat.replace(np.nan, 0) 
            fvalue_selector = SelectKBest(f_classif, k=min(Newfeat.shape[1],125))
            FX_kbest = fvalue_selector.fit_transform(Newfeat, y_train)
            bestlist=[]
            gc.collect()
            
    
            for b in (zip(fvalue_selector.get_support(),Newfeat.columns)):
                if b[0]==True and b[1]!='TARGET':
                    bestlist.append(b[1])
    
            bestfeat=Newfeat[bestlist].copy()
            
    
            gc.collect()
        X_train=pd.concat([X_train,bestfeat],axis=1) 
        for cols in bestfeat:
            print(cols)
        bestfeat=pd.DataFrame()
        if opp ==4:
            fvalue_selector = SelectKBest(f_classif, k=min(X_train.shape[1],500))
            FX_kbest = fvalue_selector.fit_transform(X_train, y_train)
            bestlist=[]
            gc.collect()
            for b in (zip(fvalue_selector.get_support(),X_train.columns)):
                if b[0]==True and b[1]!='TARGET':
                    bestlist.append(b[1])
            var=bestlist
            
    
    df=pd.concat([X_train,y_train],axis=1)
    del X_train,y_train
    df = df.rename(columns=lambda x: trim(x))
    gc.collect()
    df=df.replace(-np.Inf, 0)
    df=df.replace(np.Inf, 0)
    df=df.replace(np.nan, 0)
    utils.reducemem(df)

    # Removing empty features
    nun = df.nunique()
    empty = list(nun[nun <= 1].index)
    
    df.drop(empty, axis = 1, inplace = True)
    print('After removing empty features there are {0:d} features'.format(df.shape[1]))
    
    # Removing features with the same distribution on 0 and 1 classes
    corr = pd.DataFrame(index = ['diff', 'p'])
    ind = df[df['TARGET'].notnull()].index
    
    for c in df.columns.drop('TARGET'):
        corr[c] = utils.corr_feature_with_target(df.loc[ind, c], df.loc[ind, 'TARGET'])
    
    corr = corr.T
    corr['diff_norm'] = abs(corr['diff'] / df.mean(axis = 0))
    
    to_del_1 = corr[((corr['diff'] == 0) & (corr['p'] > .04))].index
    to_del_2 = corr[((corr['diff_norm'] < .4) & (corr['p'] > .04))].drop(to_del_1).index
    to_del = list(to_del_1) + list(to_del_2)
    if 'SK_ID_CURR' in to_del:
        to_del.remove('SK_ID_CURR')
    
    df.drop(to_del, axis = 1, inplace = True)
    print('After removing features with the same distribution on 0 and 1 classes there are {0:d} features'.format(df.shape[1]))
    
    # Removing features with not the same distribution on train and test dfsets
    corr_test = pd.DataFrame(index = ['diff', 'p'])
    target = df['TARGET'].notnull().astype(int)
    
    for c in df.columns.drop('TARGET'):
        corr_test[c] = utils.corr_feature_with_target(df[c], target)
    
    corr_test = corr_test.T
    corr_test['diff_norm'] = abs(corr_test['diff'] / df.mean(axis = 0))
    
    bad_features = corr_test[((corr_test['p'] < .04) & (corr_test['diff_norm'] > 1))].index
    bad_features = corr.loc[bad_features][corr['diff_norm'] == 0].index
    
    df.drop(bad_features, axis = 1, inplace = True)
    print('After removing features with not the same distribution on train and test dfsets there are {0:d} features'.format(df.shape[1]))
    
    del corr, corr_test
    gc.collect()
    
    # Removing features not interesting for classifier
    clf = LGBMClassifier(random_state = 0)
    train_index = df[df['TARGET'].notnull()].index
    train_columns = df.drop('TARGET', axis = 1).columns
    
    score = 1
    new_columns = []
    while score > .6:
        train_columns = train_columns.drop(new_columns)
        clf.fit(df.loc[train_index, train_columns], df.loc[train_index, 'TARGET'])
        f_imp = pd.Series(clf.feature_importances_, index = train_columns)
        score = roc_auc_score(df.loc[train_index, 'TARGET'],
                              clf.predict_proba(df.loc[train_index, train_columns])[:, 1])
        new_columns = f_imp[f_imp > 0].index
    
    df.drop(train_columns, axis = 1, inplace = True)
    
    # create a dictionnary save it and load it before loading part 3 csv

    
    df.to_csv('train//df_application_train_new2.csv',index=False)
    CTdict = {}
    for i in df.columns:
        CTdict[i] = df[i].dtypes
    
    
    
    f = open("train//ctdict2.pkl", "wb")
    pickle.dump(CTdict, f)
    f.close()
    
    return df