import numpy as np
import seqs,seqs.io 

def simple_concat(in_path1="../../AArtyk/time/train",
                  in_path2="../../AArtyk/time/train",
                  out_path="test",img_seq=True):
    read_actions=seqs.io.build_action_reader(img_seq=img_seq,as_dict=True)
    actions1=read_actions(in_path1)
    actions2=read_actions(in_path2)
#    dim=2 if img_seq else 1
    new_actions=concat_actions(actions1,actions2,dim=1)
    save_actions=seqs.io.ActionWriter(img_seq=True)  
    save_actions(new_actions,out_path)

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
    new_array=np.concatenate((array1,array2),axis=dim)
    print(new_array.shape)
    return seqs.Action(new_array,action1.name,
    	                    action1.cat,action1.person)
