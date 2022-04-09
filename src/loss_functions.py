import numpy as np

def RMSE(df_true, df_pred):
    true = df_true.values[df_true.values.nonzero()]
    pred = df_pred.values[df_true.values.nonzero()]
    return np.sqrt((((true - pred)**2).sum()) / true.size)