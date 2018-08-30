import numpy as np
import feats

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
    return list(np.sum(img_array,axis=0))

def hist_y(img_array,pcloud):
    img_array[img_array!=0]=1.0
    return list(np.sum(img_array,axis=1))

class HistZ(object):
    def __init__(self,x=False,dim=(64,256)):
        self.proj = int(x)
        self.dim=dim

    def __call__(self,img_array,pcloud):
        hist_i=np.shape(self.dim)
        for point_i in pcloud:
            x_i,z_i=int(point_i[0]),int(point_i[2])
            hist_i[x_i][z_i]+=1.0
        hist_i=np.sum(hist_i,axis=0)
        print(hist_i.shape)
        return hist_i

def count_mins(feature_i):
    extr=local_extr(feature_i)
    n_min=extr[extr==2].shape[0]
    n_max=extr[extr==(-2)].shape[0]
    return [n_min,n_max]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))