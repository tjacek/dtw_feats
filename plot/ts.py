import numpy as np
import matplotlib.pyplot as plt
import plot,utils
import seqs,seqs.select,seqs.io
import pandas as pd 

def plot_stats(in_path='mhad/simple/skew',out_path='mhad/imgs'):
    actions_by_cats=get_actions(in_path)
    utils.make_dir(out_path)
    for cat_i in actions_by_cats.keys():
        plots=get_feature_plot(actions_by_cats[cat_i])
        for j,plot_j in enumerate(plots):
            plot_path=out_path+'/'+str(cat_i)+"_"+str(j)
            print(plot_path)
            save_plot(plot_j,plot_path)

def get_actions(in_path):
    read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_action(in_path)
    actions=seqs.person_rep(actions)
    return seqs.by_cat(actions)

def get_feature_plot(actions):
    n_feats=actions[0].dim()
    features=[ action_i.as_features() for action_i in actions]
    action_names=[ action_i.name for action_i in actions]
#    features=utils.separate(features)
    features=[[feature_i[feat_index]  
                for feature_i  in features]
                    for feat_index in range(n_feats)]
    features=[ mask_features(feature_i) for feature_i in features]
    print(features[0].shape)
    return [pd.DataFrame(feature_i,columns=action_names,
                index=range(len(feature_i)))
                for feature_i in features]

def mask_features(feature_i):
    max_len=max([len(f) for f in feature_i])
    diff=[ max_len-len(f) for f in feature_i]
    mask=[ [0 for i in range(k)] for k in diff]
    mask_feat=[list(f_j)+mask_j 
                for f_j,mask_j in zip(feature_i,mask)]
    return np.array(mask_feat).T

def save_plot(plot_i,out_path_i):
    ax=plot_i.plot()
    ax.get_figure()
    plt.savefig(out_path_i)
    plt.clf()
    plt.close()