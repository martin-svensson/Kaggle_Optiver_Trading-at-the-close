# Preprocessing script
#   Based on findings in EDA

# -- Imports
#%%
import pandas as pd
import numpy as np
import pickle
pd.options.display.float_format = '{:,}'.format

# -- Read in data
#%%
df_train = pd.read_csv("../data/train.csv")

# -- Reduce memory requirements
#%%
df_train_floats = df_train.select_dtypes(np.float64).drop("target", axis = 1) # keep target in 64-bit
df_train_ints = df_train.select_dtypes(np.int64)

df_train[df_train_floats.columns] = df_train_floats.astype(np.float16)
df_train[df_train_ints.columns] = df_train_ints.astype(np.int16)

# -- data split
#%%
df_train["max_time"] = df_train.groupby("stock_id")["time_id"].transform("max")
df_training = df_train.loc[df_train["time_id"] < df_train["max_time"] * 0.8]
df_testing = df_train.loc[~df_train["row_id"].isin(df_training["row_id"])]

df_training = df_training.drop("max_time", axis = 1)
df_testing = df_testing.drop("max_time", axis = 1)

# -- Save preprocessed data
#%%
out = {
    "df_training": df_training,
    "df_testing": df_testing
}

with open('../output/preprocessing.pickle', 'wb') as f:
    pickle.dump(out, f)
# %%
