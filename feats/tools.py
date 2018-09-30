import feats,seqs.io
import utils,cv2
import feats.local
from feats.glob import *
import scipy.stats

def hough_feats():
    return feats.GlobalFeatures(HoughDispersion())

def stats_feats():
    raw_funs=[np.mean,np.std,scipy.stats.skew]
    return feats.GlobalFeatures(raw_funs)

def extrem_feats():
    glob_feats=SeriesFeature(extrm_count)
    return feats.GlobalFeatures(glob_feats)

def get_histogram_feats( extr=False):
    raw_hist=[hist_x,hist_y,HistZ(0,1),HistZ(1,1)]
    return [smooth_feat(hist_i,extr) for hist_i in raw_hist]	

def smooth_feat(feat_i,extr=False):
    fun_list=[feat_i,feats.FourierSmooth()]
    if(extr):
        fun_list.append(feats.local.count_mins)
    return feats.FeatPipeline(fun_list)