import numpy as np
from sklearn.metrics import mean_squared_error

def RMSE(true, pred):
    return np.sqrt(mean_squared_error(y_true=true, y_pred=pred))