# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 00:20:20 2020

@author: Jean-Philippe
"""
import numpy as np   # import numpy
import pandas as pd 
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import ranksums
from bayes_opt import BayesianOptimization
import lightgbm as gbm
import types
import gc   # for gabage collection
import seaborn as sns  # data visualization lib
import time
import matplotlib.pyplot as plt
import glob 
import config


warnings.simplefilter(action = 'ignore', category = FutureWarning)
def cv_scores(df, num_folds, params, stratified = False, verbose = -1, 
              save_train_prediction = False, train_prediction_file_name = 'train_prediction.csv',
              save_test_prediction = True, test_prediction_file_name = 'test_prediction.csv'):
    warnings.simplefilter('ignore')
    
    clf = LGBMClassifier(**params)

    # Divide in training/validation and test data
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)
        
    # Create arrays and dataframes to store results
    train_pred = np.zeros(train_df.shape[0])
    train_pred_proba = np.zeros(train_df.shape[0])

    test_pred = np.zeros(train_df.shape[0])
    test_pred_proba = np.zeros(train_df.shape[0])
    
    prediction = np.zeros(test_df.shape[0])
    
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    df_feature_importance = pd.DataFrame(index = feats)
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        print('Fold', n_fold, 'started at', time.ctime())
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        clf.fit(train_x, train_y, 
                eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                verbose = verbose, early_stopping_rounds = 500)

        train_pred[train_idx] = clf.predict(train_x, num_iteration = clf.best_iteration_)
        train_pred_proba[train_idx] = clf.predict_proba(train_x, num_iteration = clf.best_iteration_)[:, 1]
        test_pred[valid_idx] = clf.predict(valid_x, num_iteration = clf.best_iteration_)
        test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
        
        prediction += \
                clf.predict_proba(test_df[feats], num_iteration = clf.best_iteration_)[:, 1] / folds.n_splits

        df_feature_importance[n_fold] = pd.Series(clf.feature_importances_, index = feats)
        
        print('Fold %2d AUC : %.6f' % (n_fold, roc_auc_score(valid_y, test_pred_proba[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()

    roc_auc_train = roc_auc_score(train_df['TARGET'], train_pred_proba)
    precision_train = precision_score(train_df['TARGET'], train_pred, average = None)
    recall_train = recall_score(train_df['TARGET'], train_pred, average = None)
    
    roc_auc_test = roc_auc_score(train_df['TARGET'], test_pred_proba)
    precision_test = precision_score(train_df['TARGET'], test_pred, average = None)
    recall_test = recall_score(train_df['TARGET'], test_pred, average = None)
    print('Full AUC score %.6f' % roc_auc_test)
    
    df_feature_importance.fillna(0, inplace = True)
    df_feature_importance['mean'] = df_feature_importance.mean(axis = 1)
    
    # Write prediction files
    if save_train_prediction:
        df_prediction = train_df[['SK_ID_CURR', 'TARGET']]
        df_prediction['Prediction'] = test_pred_proba
        df_prediction.to_csv(train_prediction_file_name, index = False)
        del df_prediction
        gc.collect()

    if save_test_prediction:
        df_prediction = test_df[['SK_ID_CURR']]
        df_prediction['TARGET'] = prediction
        df_prediction.to_csv(test_prediction_file_name, index = False)
        del df_prediction
        gc.collect()
    
    return df_feature_importance, \
           [roc_auc_train, roc_auc_test,
            precision_train[0], precision_test[0], precision_train[1], precision_test[1],
            recall_train[0], recall_test[0], recall_train[1], recall_test[1], 0]

def display_folds_importances(feature_importance_df_, n_folds = 5):
    n_columns = 3
    n_rows = (n_folds + 1) // n_columns
    _, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 8 * n_rows))
    for i in range(n_folds):
        sns.barplot(x = i, y = 'index', data = feature_importance_df_.reset_index().sort_values(i, ascending = False).head(20), 
                    ax = axes[i // n_columns, i % n_columns])
    sns.barplot(x = 'mean', y = 'index', data = feature_importance_df_.reset_index().sort_values('mean', ascending = False).head(20), 
                    ax = axes[n_rows - 1, n_columns - 1])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.show()



def Train(df,BO):
    
    def lgbm_evaluate(**params):
        warnings.simplefilter('ignore')
      
        params['num_leaves'] = int(params['num_leaves'])
        params['max_depth'] = int(params['max_depth'])
        params['bagging_freq'] = int(params['bagging_freq'])
        params['max_bin'] = int(params['max_bin'])
        params['n_estimators'] = int(params['n_estimators'])
        
        clf = LGBMClassifier(**params, nthread = 4)
        train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)
    
    
        folds = KFold(n_splits = 2, shuffle = True, random_state = 1001)
            
        test_pred_proba = np.zeros(train_df.shape[0])
        
        feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
        
        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
            train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
            valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
    
            clf.fit(train_x, train_y, 
                    eval_set = [(train_x, train_y), (valid_x, valid_y)], eval_metric = 'auc', 
                    verbose = False, early_stopping_rounds = 100)
    
            test_pred_proba[valid_idx] = clf.predict_proba(valid_x, num_iteration = clf.best_iteration_)[:, 1]
            
            del train_x, train_y, valid_x, valid_y
            gc.collect()

        return roc_auc_score(train_df['TARGET'], test_pred_proba)
    
    scores_index = [
    'roc_auc_train', 'roc_auc_test',
    'precision_train_0', 'precision_test_0',
    'precision_train_1', 'precision_test_1',
    'recall_train_0', 'recall_test_0',
    'recall_train_1', 'recall_test_1',
    'LB']
    scores = pd.DataFrame(index = scores_index)
    warnings.simplefilter(action = 'ignore', category = FutureWarning)
    gbm.Booster.feature_name
    gbm.Booster.feature_name = df.columns
    gbm.Booster.feature_name = types.MethodType(lambda self: gbm.Booster.feature_name, gbm.Booster)
    gc.collect()
    if BO =='1':
        bo = BayesianOptimization(lgbm_evaluate, config.params)
        bo.maximize(init_points = 5 , n_iter =5)
        best_params = bo.max['params']
        best_params['num_leaves'] = int(best_params['num_leaves'])
        best_params['max_depth'] = int(best_params['max_depth'])
        best_params['bagging_freq'] = int(best_params['bagging_freq'])
        best_params['max_bin'] = int(best_params['max_bin'])
        best_params['n_estimators'] = int(best_params['n_estimators'])
        feature_importance, scor = cv_scores(df,5 , best_params, test_prediction_file_name = 'prediction_1.csv')
    else:
        feature_importance, scor = cv_scores(df,5 , config.lgbm_params, test_prediction_file_name = 'prediction_1.csv')
    step = 'Bayesian optimization for new set'
    scores[step] = scor
    scores.loc['LB', step] = .797
    scores.T
    print(display_folds_importances(feature_importance))
    print(feature_importance.sort_values('mean', ascending = False).head(20))
    