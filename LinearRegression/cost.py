import numpy as np

def MSE(y_pred, y_true):
    #Checking the predicted labels and real labels are the same length 
    if y_pred.shape[0] != y_true.shape[0]:
        return False
    
    p = y_pred.shape[0]
    y_diff = y_pred - y_true

    return (1 / p) * y_diff.dot(y_diff)


    
