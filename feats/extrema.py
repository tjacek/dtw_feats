import numpy as np 

def count_mins(feature_i):
    extr=local_extr(feature_i)
    n_min= get_location(extr,2)[0].shape[0]
    n_max= get_location(extr,-2)[0].shape[0]
    return [n_min,n_max]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))

def get_location(array_i,value):
    return np.where(array_i==value)