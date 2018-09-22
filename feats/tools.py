import feats,seqs.io
import utils,cv2
from feats.local import *
from feats.glob import *
import scipy.stats

class ActionImgs(object):        
    def __call__(in_path,out_path):#,local_feats):
        read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
        actions=read_actions(in_path)
        utils.make_dir(out_path)
        for action_i in new_actions:
            out_i=out_path+'/'+action_i.name+".png"
            cv2.imwrite(out_i,self.get_img(action_i))

class TimeActionImgs(object):
    def __init__(self,local_feats):
        if(type(local_feats)!=feats.LocalFeatures):
            local_feats=feats.LocalFeatures(local_feats)
        self.local_feats=local_feats

    def get_img(self,action_i):
        dummy_action=action_i(self.local_feats,whole_seq=False) 
        return dummy_action.as_array()

#def timeless_img(action_i):
#    features=action_i.as_features()
#    x,y=features[0],features[1]

def stats_feats():
    raw_funs=[np.mean,np.std,scipy.stats.skew]
    return feats.GlobalFeatures([GlobalExtractor(fun_i) 
                    for fun_i in raw_funs])

def quality_feats():
    local_feats=[GlobalExtractor(np.amax),
                 GlobalExtractor(np.median),GlobalExtractor(np.min)]
    return feats.GlobalFeatures(local_feats)

def get_histogram_feats( extr=False):
    raw_hist=[hist_x,hist_y,HistZ(0,1),HistZ(1,1)]
    return [smooth_feat(hist_i,extr) for hist_i in raw_hist]	

def smooth_feat(feat_i,extr=False):
    fun_list=[feat_i,feats.FourierSmooth()]
    if(extr):
        fun_list.append(feats.local.count_mins)
    return feats.FeatPipeline(fun_list)