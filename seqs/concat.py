import numpy as np
import seqs,seqs.io 

def apply_concat(in_path1='seqs/max_z',in_path2='seqs/all',out_path='seqs/full'):
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=True)
    actions1=read_actions(in_path1)
    actions2=read_actions(in_path2)
    unified_actions=concat_actions(actions1,actions2)
    save_actions=actions.io.ActionWriter(img_seq=False)
    save_actions(unified_actions,out_path) 

def concat_actions(actions1,actions2,dim=1): 
    names=actions1.keys()
    return [ unify_actions(actions1[name_i],actions2[name_i],dim) 
                        for name_i in names]

def unify_actions(action1,action2,dim=1):
    array1=action1.as_array()
    array2=action2.as_array()
    if(array1.shape[0]!=array2.shape[0]):
        new_dim=min(array1.shape[0],array2.shape[0])
        array1=array1[:new_dim]
        array2=array2[:new_dim]
    print(array1.shape)
    print(array2.shape)
    new_array=np.concatenate((array1,array2),axis=dim)
    print(new_array.shape)
    return seqs.Action(new_array,action1.name,
    	                    action1.cat,action1.person)