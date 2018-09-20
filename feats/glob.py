import numpy as np
import feats,feats.preproc
import scipy.stats

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