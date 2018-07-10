import numpy as np
import scipy.stats
import instances

class GlobalFeatures(object):
    def __init__(self):
        self.feature_extractor=[avg,std,skew]

    def __call__(self,action_i):
        series=action_i.as_feature()
        global_feats=[]
        for serie_i in series:
            for extractor_i in self.feature_extractor:
                global_feats.append(extractor_i(serie_i))
        return instances.Instance(global_feats,action_i.cat,
                            action_i.person,action_i.name) 

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