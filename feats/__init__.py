import numpy as np
import scipy.stats
import dataset.instances,seqs.io

class Features(object):
    def __init__(self,feature_extractors):
        if(type(feature_extractors)!=list):
            feature_extractors=[feature_extractors]
        self.feature_extractors=feature_extractors

class GlobalFeatures(Features):
    def __call__(self,action_i):
#        img_i=action_i.as_array()
        global_feats=[]
        for extractor_j in self.feature_extractors:
            global_feats+=extractor_j(action_i)
        return dataset.instances.Instance(global_feats,action_i.cat,
                            action_i.person,action_i.name) 

    def apply(self,in_path='mra/seqs/all',out_path='mra/simple/basic.txt'):
        read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_action(in_path)
        insts=dataset.instances.InstsGroup([self(action_i) 
                    for action_i in actions])
        insts.to_txt(out_path)

class LocalFeatures(Features):
    def __call__(self,img_i):
        img_i=preproc_img(img_i)
        point_cloud=extract_points(img_i)
        features=[]
        for extractor_j in self.feature_extractors:
            features+=extractor_j(img_i,point_cloud)
        return features
    
    def apply(self,in_path,out_path):
        seqs.io.transform_actions(in_path,out_path,self,
            img_in=True,img_out=False,whole_seq=False)

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

    def __call__(self,img_array,action_array):
        result=self.functions[0](img_array,action_array)
        for fun_i in self.functions[1:]:
            result=fun_i(result)
        return list(result)