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

def corl(img_array,pcloud):
    def corl_helper(i,j):
        return scipy.stats.pearsonr(pcloud[:,i],pcloud[:,j])[0]
    return [corl_helper(0,1) ,corl_helper(0,2),corl_helper(1,2)]

def hist_x(img_array,pcloud):
    img_array[img_array!=0]=1.0
    return list(np.sum(img_array,axis=0))

def hist_y(img_array,pcloud):
    img_array[img_array!=0]=1.0
    return list(np.sum(img_array,axis=1))

class HistZ(object):
    def __init__(self,x=False,sum_axis=0,dim=(65,256)):
        self.proj = int(x)
        self.dim=dim
        self.sum_axis=sum_axis

    def __call__(self,img_array,pcloud):
        hist_i=np.zeros(self.dim)
        for point_i in pcloud:
            x_i,z_i=int(point_i[self.proj]),int(point_i[2])
            hist_i[x_i][z_i]+=1.0
        return np.sum(hist_i,axis=self.sum_axis)

def count_mins(feature_i):
    extr=local_extr(feature_i)
    n_min=extr[extr==2].shape[0]
    n_max=extr[extr==(-2)].shape[0]
    return [n_min,n_max]

def local_extr(feature_i):
    diff_i=np.diff(feature_i)
    return np.diff( np.sign(diff_i))