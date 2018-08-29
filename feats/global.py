import numpy as np,feats

def corl(img_array,pcloud):
    def corl_helper(i,j):
        return scipy.stats.pearsonr(pcloud[:,i],pcloud[:,j])[0]
    return [corl_helper(0,1) ,corl_helper(0,2),corl_helper(1,2)]

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

def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0][1]