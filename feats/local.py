import numpy as np
import cv2
import feats,feats.preproc,utils,seqs.io 

def avg(img_array,action_array):
    return list(np.mean(action_array,axis=0))

def std(img_array,pcloud):
    return list(np.std(pcloud,axis=0))

def skew(img_array,pcloud):
    return list(scipy.stats.skew(pcloud,axis=0))

def area(img_array,point_cloud):
    n_points=point_cloud.shape[0]
    size=float(img_array.shape[0] * img_array.shape[1])
    return [n_points/size]

def hist_x(img_array,pcloud):
    img_array[img_array!=0]=1.0
    fs=feats.FourierSmooth()
    return list(fs(np.sum(img_array,axis=0)))

def hist_y(img_array,pcloud):
    img_array[img_array!=0]=1.0
    return list(fs(np.sum(img_array,axis=1)))

def count_mins(feature_i):
    extr=local_extr(feature_i)
    n_min=extr[extr==2].shape[0]
    n_max=extr[extr==(-2)].shape[0]
    return [n_min,n_max]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))
