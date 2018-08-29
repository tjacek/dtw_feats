import numpy as np
import scipy.stats
import dataset.instances,seqs.io

class LocalFeatures(object):
    def __init__(self, feature_extractors):
        if(type(feature_extractors)!=list):
            feature_extractors=[feature_extractors]
        self.feature_extractors=feature_extractors

    def __call__(self,img_i):
        print(np.amax(img_i))
        img_i=preproc_img(img_i)
        point_cloud=extract_points(img_i)
        features=[]
        for extractor_j in self.feature_extractors:
            features+=extractor_j(img_i,point_cloud)
        return features

class GlobalFeatures(object):
    def __init__(self,feats):
        if(type(feats)!=list):
            feats=[feats]
        self.feature_extractor=feats#[avg,std,skew]

    def __call__(self,action_i):
        features_i=action_i.as_features()
        global_feats=[]
        for extractor_k in self.feature_extractor:
            for feature_j in features_i:
                fusion_jk=extractor_k(None,feature_j)
                if(type(fusion_jk)==list):
                    global_feats+=fusion_jk
                else:
                    global_feats.append(fusion_jk)
        return dataset.instances.Instance(global_feats,action_i.cat,
                            action_i.person,action_i.name) 

    def make_dataset(self,in_path='mra/seqs/all',out_path='mra/simple/basic.txt'):
        read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_action(in_path)
        insts=dataset.instances.InstsGroup([self(action_i) 
                    for action_i in actions])
        insts.to_txt(out_path)

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

def corl(img_array,pcloud):
    def corl_helper(i,j):
        return scipy.stats.pearsonr(pcloud[:,i],pcloud[:,j])[0]
    return [corl_helper(0,1) ,corl_helper(0,2),corl_helper(1,2)]

def area(img_array,point_cloud):
    n_points=point_cloud.shape[0]
    size=float(img_array.shape[0] * img_array.shape[1])
    return [n_points/size]

def rapid_change(dummy,feature_i):
    feature_i=feature_i.astype(float)
    feature_i=np.diff(feature_i)
    min_i=np.amin(feature_i)
    max_j=np.amax(feature_i)
    feature_i=(feature_i-min_i)/max_j
    feature_i*feature_i
    return np.mean(feature_i)

def autocorl_feat(dummy,feature_i):
    diff_i=np.diff(feature_i)
    return np.mean([autocorr(diff_i,j) 
                for j in range(1,len(feature_i)-2)])

def extr_feat(dummy, feature_i):
    diff_i=np.diff(feature_i)
    extr_i=np.diff( np.sign(diff_i))
    return [count_values(extr_i,2.0),count_values(extr_i,-2.0)]

def count_values(extr_i,value=2.0):
    min_i=np.copy(extr_i)
    min_i[min_i!=value]=0.0
    min_i[min_i==value]=1.0
    return np.sum(min_i,axis=0) 

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0][1]