import numpy as np
import cv2
import feats,utils,seqs.io 

class FeatPipeline(object):
    def __init__(self, functions):
    	self.functions=functions

    def __call__(img_array,action_array):
        for fun_i in self.functions:
        	img_array=fun_i(img_array)
		return img_array

def action_imgs(in_path,out_path,local_feats):
    if(type(local_feats)!=feats.LocalFeatures):
        local_feats=feats.LocalFeatures(local_feats)
    read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
    actions=read_actions(in_path)
    new_actions=[action_i(local_feats,whole_seq=False) 
                    for action_i in actions]
    utils.make_dir(out_path)
    for action_i in new_actions:
        out_i=out_path+'/'+action_i.name+".png"
        img_i=action_i.as_array()
        cv2.imwrite(out_i,img_i)

def avg(img_array,action_array):
    return list(np.mean(action_array,axis=0))

def std(img_array,pcloud):
    return list(np.std(pcloud,axis=0))

def skew(img_array,pcloud):
    return list(scipy.stats.skew(pcloud,axis=0))

def hist_x(img_array,pcloud):
    img_array[img_array!=0]=1.0
    return list(np.sum(img_array,axis=0))

def count_mins(feature_i):
    extr=local_extr(feature_i)
    n_min=extr[extr==2].shape[0]
    n_max=extr[extr==(-2)].shape[0]
    return [n_min,n_max]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))
