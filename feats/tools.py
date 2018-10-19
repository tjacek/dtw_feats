import feats,feats.preproc,feats.extrema
import utils,cv2,seqs.io
from feats.local import *
from feats.glob import *

def basic_feats():
    stat_funs=[np.std,scipy.stats.skew]
    all_feats=[area,corl]
    for stat_i in stat_funs:
        all_feats+=[ AxisFeat(stat_i,axis_i) 
                        for axis_i in range(3)]
    return feats.LocalFeatures(all_feats)

def stats_feats():
    raw_funs=series_decorator([np.mean,np.std,scipy.stats.skew])
    return feats.GlobalFeatures(raw_funs)

def optim_feats():
    glob_feats=series_decorator([optim_position,extrm_count])
    return feats.GlobalFeatures(glob_feats)

def smooth_feats():
    glob_feats=[SeriesFeature(local_smoothnes)] 
    return feats.GlobalFeatures(glob_feats)

def hough_feats():
    return feats.GlobalFeatures(HoughDispersion())

def histogram_feats( extr=True):
    raw_hist=[hist_x,hist_y,HistZ(0,1),HistZ(1,1)]
    local_feats=[smooth_feat(hist_i,extr) for hist_i in raw_hist]	
    return feats.LocalFeatures(local_feats)

def smooth_feat(feat_i,extr=False):
    fun_list=[feat_i,feats.preproc.FourierSmooth()]
    if(extr):
        fun_list.append(feats.extrema.count_mins)
    return feats.FeatPipeline(fun_list)

def series_decorator(raw_feats):
    return [ SeriesFeature(feat_i) for feat_i in raw_feats]