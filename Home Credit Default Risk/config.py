"""
all parameters and configurations
"""
import pandas as pd


DATA_PATH = ""
n_estimator = 100
max_depth = 10
seed = 10
n_jobs = -1
stop_rounds = 1000
k_folds = 5
verbose = 500


lgbm_params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'nthread': 4,
            'n_estimators': 50000,
            'learning_rate': .0125,
            'num_leaves': 200,
            'colsample_bytree': .9497036,
            'subsample': .8715623,
            'max_depth': 15,
            'reg_alpha': .041545473,
            'reg_lambda': .0735294,
            'min_split_gain': .0222415,
            'min_child_weight': 39.3259775,
            "max_bin": 1000,
            'silent': -1,
            'verbose': -1
}


params = {'colsample_bytree': (0.75, 1),
          'learning_rate': (.005, .025), 
          'num_leaves': (25, 200), 
          'subsample': (0.75, 1), 
          'max_depth': (10, 20), 
          'reg_alpha': (.025, .05), 
          'reg_lambda': (.055, .08), 
          'min_split_gain': (.01, .05),
          'min_child_weight': (20, 60),
          'max_bin': (100,2000),
          'n_estimators': (5000,50000),
          'feature_fraction':( 0.5,1),
          'bagging_fraction':(0.5,1),
          'bagging_freq': (3,10),
         }