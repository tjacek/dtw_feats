import numpy as np
import feats,feats.preproc,feats.extrema
import scipy.stats
from feats.action_imgs import TimelessActionImgs
from sklearn.linear_model import LinearRegression

class SeriesFeature(object):
    def __init__(self,fun):
        self.fun=fun

    def __call__(self,action_i):
        all_feats=[]
        for feat_i in action_i.as_features():
            result=self.fun(feat_i)
            if(type(result)==list):
                all_feats+=result
            else:
                all_feats.append(result)
        print(all_feats)
        return all_feats                  

class HoughDispersion(object):
    def __init__(self,size=(65,65),indexes=(0,1),blur=3):
        self.action_img=TimelessActionImgs(size=size,indexes=indexes,blur=blur,hough="ellipse")

    def __call__(self,action_i):
        img_i,size=self.action_img.get_img(action_i)
        img_i=img_i.astype(float)
        img_i/=size#np.sum(img_i)
        return [np.max(img_i),np.median(img_i),np.std(img_i)]

def extrm_count(feature_i):
    smooth_feat=feats.preproc.FourierSmooth()(feature_i)
    return feats.extrema.count_mins(smooth_feat)

def optim_position(feature_i):
    min_i=feats.extrema.relative_location(feature_i,np.amax(feature_i))
    max_i=feats.extrema.relative_location(feature_i,np.amin(feature_i))
    return [min_i,max_i]

def local_smoothnes(feature_i):
    feature_i=feats.preproc.FourierSmooth()(feature_i)
    pos_i=feats.extrema.get_location(feature_i)
    windows=feats.extrema.get_window(pos_i,feature_i,k=5)
    return np.amax([total_smoothnes(window_i) 
                    for window_i in windows])

def total_smoothnes(feature_i):
    feature_i-= np.min(feature_i)-1
    diff_i=np.diff(feature_i)
    size=float(feature_i.shape[0])
    total_smooth=np.sum(diff_i/feature_i[:-1])
    return total_smooth/size

#def b(feature_i):

def freq_skewnes(feature_i):
    magnitude=feats.preproc.fourier_magnitude(feature_i)
    print(magnitude.shape)
    return list(scipy.stats.skew(magnitude))

def rapid_change(feature_i):
    feature_i=feature_i.astype(float)
    feature_i=np.diff(feature_i)
    feature_i*feature_i
    return np.mean(feature_i)

def autocorl_feat(feature_i):
    diff_i= standarize(np.diff(feature_i))
    return np.mean([autocorr(diff_i,j) 
                for j in range(1,len(feature_i)-2)])

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0][1]

def standarize(feature_i):
    min_i=np.amin(feature_i)
    max_j=np.amax(feature_i)
    feature_i=(feature_i-min_i)/max_j