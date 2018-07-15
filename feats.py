import numpy as np
import scipy.stats
import instances,seqs.io

class LocalFeatures(object):
    def __init__(self, feature_extractors):
        self.feature_extractors=feature_extractors

    def __call__(self,img_i):
        img_i=preproc_img(img_i)
        point_cloud=extract_points(img_i)
        features=[]
        for extractor_j in self.feature_extractors:
            features+=extractor_j(img_i,point_cloud)
        return features

class BasicFeatures(LocalFeatures):
    def __init__(self):
        all_features=[area]
        super(BasicFeatures, self).__init__(all_features)  

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

def preproc_img(img_i,img_size=64):
    return img_i[:img_size]

def extract_points(img_i):
    points=[]
    for (x, y), depth in np.ndenumerate(img_i):
        if(depth>0):
            point_d=np.array([x,y,depth])
            points.append(point_d)
    return np.array(points)

def avg(action_array):
    return np.mean(action_array,axis=0)

def std(action_array):
    return np.std(action_array,axis=0)

def skew(action_array):
    return scipy.stats.skew(action_array,axis=0)

def area(img_array,point_cloud):
    n_points=point_cloud.shape[0]
    size=float(img_array.shape[0] * img_array.shape[1])
    return [n_points/size]

if __name__ == "__main__":
    transform=BasicFeatures()
    seqs.io.transform_actions(in_path='mhad/time',out_path='mhad/all',transform=transform)