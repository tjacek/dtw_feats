import feats,seqs.io
import utils,cv2
from feats.local import *
from feats.glob import *

def action_imgs(in_path,out_path,local_feats):
    if(type(local_feats)!=feats.LocalFeatures):
        local_feats=feats.LocalFeatures(local_feats)
    read_actions=seqs.io.build_action_reader(img_seq=True,as_dict=False)
    actions=read_actions(in_path)
    new_actions=[action_i(local_feats,whole_seq=False) 
                    for action_i in actions]
    utils.make_dir(out_path)
    for action_i in new_actions:
        out_i=out_path+'/'+action_i.name+".png"
        img_i=action_i.as_array()
        cv2.imwrite(out_i,img_i)

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