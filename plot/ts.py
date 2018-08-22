import numpy as np
import matplotlib.pyplot as plt
import plot,utils
import seqs,seqs.select,seqs.io
import pandas as pd 

def plot_stats(in_path='mhad/simple/skew',out_path='mhad/imgs'):
    read_action=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_action(in_path)
    print(len(actions))
    actions=seqs.select.select(actions,1)
    print(len(actions))
    actions_by_cats=seqs.by_cat(actions)
    utils.make_dir(out_path)
    for cat_i in actions_by_cats.keys():
        plots=get_feature_plot(actions_by_cats[cat_i])
        for j,plot_j in enumerate(plots):
            plot_path=out_path+'/'+str(cat_i)+"_"+str(j)
            print(plot_path)
            save_plot(plot_j,plot_path)
    
def get_feature_plot(actions):
    features=[ action_i.as_feature() for action_i in actions]
    features=utils.separate(features)
    features=[ mask_features(feature_i) for feature_i in features]
    print(features[0].shape)
    return [pd.DataFrame(feature_i,index=range(len(feature_i)))
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