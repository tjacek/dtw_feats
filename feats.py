import numpy as np
import scipy.stats

class GlobalFeatures(object):
    def __init__(self):
        self.feature_extractor=[avg,std,skew]

    def __call__(self,action_i):
        action_array=action_i.as_array()
        features=[]
        for extractor_i in self.feature_extractor:
            features+=list(extractor_i(action_array))
        return features

def avg(action_array):
    return np.mean(action_array,axis=0)

def std(action_array):
    return np.std(action_array,axis=0)

def skew(action_array):
    return scipy.stats.skew(action_array,axis=0)

def area(img_array):
    img_array[img_array!=0]=1.0
    size=float(img_array.shape[0] * img_array.shape[1])
    return np.sum(img_array)