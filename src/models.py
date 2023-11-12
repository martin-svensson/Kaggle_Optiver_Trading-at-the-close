# Description
#   This script applies feature engineering, defines model specifications, fits models (combination of feature engineering and model specifications) 
#   and evaluates the models on the test set (df_testing)

# -- Imports and data
#%%
import numpy as np
import pandas as pd
import pickle
import lightgbm
import sklearn as sk
from sklearn import ensemble
from feature_engineering import calculate_imbalance_features

with open('../output/preprocessing.pickle', 'rb') as f:
    data = pickle.load(f)

# -- Model specifications
#%%
lightgbm_spec = lightgbm.LGBMRegressor(
    boosting_type = 'gbdt', 
    num_leaves = 31, 
    max_depth = -1, 
    learning_rate = 0.1, 
    n_estimators = 100, 
    subsample_for_bin = 200000, 
    objective = 'mae', 
    class_weight = None, 
    min_split_gain = 0.0, 
    min_child_weight = 0.001, 
    min_child_samples = 20, 
    subsample = 1.0, 
    subsample_freq = 0, 
    colsample_bytree = 1.0, 
    reg_alpha = 0.0, 
    reg_lambda = 0.0, 
    random_state = 9347, 
    n_jobs = -1, 
    importance_type = 'split',
    force_row_wise = True
)

xgboost_spec = ensemble.GradientBoostingRegressor(
    n_estimators=2000,
    validation_fraction=0.2, # early stopping this way makes a random split, which is not ideal for timeseries data 
    n_iter_no_change=50,
    tol=0.001,
    learning_rate=0.005,
    loss="absolute_error",
    random_state=8193,
    verbose=1
)

# -- Apply feature engineering
#%%
df_training = calculate_imbalance_features(data["df_training"])
df_testing = calculate_imbalance_features(data["df_training"])

#%%
df_training_X = df_training.query("target.notna()").drop(["row_id", "time_id"], axis = 1)
df_training_y = df_training.query("target.notna()")["target"]
df_testing_X = df_testing.query("target.notna()").drop(["row_id", "time_id"], axis = 1)
df_testing_y = df_testing.query("target.notna()")["target"]

# -- Fit models
#%%
lightgbm_spec.fit(df_training_X, df_training_y)
#%%
xgboost_spec.fit(df_training_X, df_training_y)

# -- Evaluate models
#%%
sk.metrics.mean_absolute_error(lightgbm_spec.predict(df_testing_X), df_testing_y)