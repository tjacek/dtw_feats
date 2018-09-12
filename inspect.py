import numpy as np
import seqs,seqs.io,seqs.select 

def show_stats(in_path,img_seq=False):
    actions=get_actions(in_path,img_seq=img_seq,select=1)
    n_actions=len(actions)
    actions_by_cat=seqs.by_cat(actions)
    cats=[ len(cat_i) for cat_i in actions_by_cat.values()]
    print(np.median(cats))

def get_actions(in_path,img_seq=False,select=False):
    read_actions=seqs.io.build_action_reader(img_seq=img_seq,as_dict=False)
    actions=read_actions(in_path)
    if(select):
        actions=seqs.select.select(actions,selector=select)
    return actions

show_stats(in_path='mra/seqs/corl',img_seq=False)