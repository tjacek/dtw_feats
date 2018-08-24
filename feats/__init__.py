import numpy as np
import scipy.stats
import dataset.instances

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

class GlobalFeatures(object):
    def __init__(self):
        self.feature_extractor=[avg,std,skew]

    def __call__(self,action_i):
        array_i=action_i.as_array()
        global_feats=[]
        for extractor_j in self.feature_extractor:
            global_feats+=extractor_j(None,array_i)
        return dataset.instances.Instance(global_feats,action_i.cat,
                            action_i.person,action_i.name) 

def basic_features():
    return LocalFeatures([area,corl,std,skew])  

def preproc_img(img_i,img_size=64):
    return img_i[:img_size]

def extract_points(img_i):
    points=[]
    for (x, y), depth in np.ndenumerate(img_i):
        if(depth>0):
            point_d=np.array([x,y,depth])
            points.append(point_d)
    return np.array(points)

def avg(img_array,action_array):
    return list(np.mean(action_array,axis=0))

def std(img_array,pcloud):
    return list(np.std(pcloud,axis=0))

def skew(img_array,pcloud):
    return list(scipy.stats.skew(pcloud,axis=0))

def corl(img_array,pcloud):
    def corl_helper(i,j):
        return scipy.stats.pearsonr(pcloud[:,i],pcloud[:,j])[0]
    return [corl_helper(0,1) ,corl_helper(0,2),corl_helper(1,2)]

def area(img_array,point_cloud):
    n_points=point_cloud.shape[0]
    size=float(img_array.shape[0] * img_array.shape[1])
    return [n_points/size]

def make_stats_feat(in_path='mra/seqs/all',out_path='mra/simple/basic.txt'):
    read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_action(in_path)
    feat_extractor=feats.GlobalFeatures()
    insts=[feat_extractor(action_i) for action_i in actions]
    dataset.instances.to_txt(out_path,insts)