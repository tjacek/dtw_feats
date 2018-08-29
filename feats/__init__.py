import numpy as np
import scipy.stats
import dataset.instances,seqs.io
import feats.local,feats.global

class Features(object):
    def __init__(self,feature_extractor):
        if(type(feature_extractor)!=list):
            feature_extractor=[feature_extractor]
        self.feature_extractor=feats

class GlobalFeatures(Features):
    def __call__(self,action_i):
        features_i=action_i.as_features()
        global_feats=[]
        for extractor_k in self.feature_extractor:
            GlobalFeatures+=extractor_k(None,features_i)
        return dataset.instances.Instance(global_feats,action_i.cat,
                            action_i.person,action_i.name) 

    def make_dataset(self,in_path='mra/seqs/all',out_path='mra/simple/basic.txt'):
        read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_action(in_path)
        insts=dataset.instances.InstsGroup([self(action_i) 
                    for action_i in actions])
        insts.to_txt(out_path)

class LocalFeatures(Features):
    def __call__(self,img_i):
        print(np.amax(img_i))
        img_i=preproc_img(img_i)
        point_cloud=extract_points(img_i)
        features=[]
        for extractor_j in self.feature_extractors:
            features+=extractor_j(img_i,point_cloud)
        return features

def preproc_img(img_i,img_size=64):
    return img_i[:img_size]

def extract_points(img_i):
    points=[]
    for (x, y), depth in np.ndenumerate(img_i):
        if(depth>0):
            point_d=np.array([x,y,depth])
            points.append(point_d)
    return np.array(points)

class FeatPipeline(object):
    def __init__(self, functions):
        self.functions=functions

    def __call__(img_array,action_array):
        for fun_i in self.functions:
            img_array=fun_i(img_array)
        return list(img_array)

class FourierSmooth(object):
    def __init__(self, n=5):
        self.n = n

    def __call__(self,feature_i):
        rft = np.fft.rfft(feature_i)
        rft[self.n:] = 0
        return np.fft.irfft(rft)