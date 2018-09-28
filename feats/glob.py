import numpy as np
import feats,feats.preproc
import scipy.stats
from feats.action_imgs import TimelessActionImgs

class HoughDispersion(object):
    def __init__(self,size=(65,65),indexes=(0,1),blur=3):
        self.action_img=TimelessActionImgs(size=size,indexes=indexes,blur=blur,hough=True)

    def __call__(self,action_i):
        img_i=self.action_img.get_img(action_i)
        #img_i=img_i.astype(float)
        #img_i/=np.sum(img_i)
        return [np.max(img_i)]
#        return [np.std(img_i)]  

def freq_skewnes(feature_i):
    magnitude=feats.preproc.fourier_magnitude(feature_i)
    print(magnitude.shape)
    return list(scipy.stats.skew(magnitude))

def simple_smoothnes(feature_i):
    return np.sd(np.diff(feature_i))/ np.mean(feature_i)

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