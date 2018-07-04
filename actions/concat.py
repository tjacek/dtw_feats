import numpy as np
import actions

def concat_actions(actions1,actions2): 
    names=actions1.keys()
    return [ unify_actions(actions1[name_i],actions2[name_i]) 
                        for name_i in names]

def unify_actions(action1,action2):
    array1=action1.as_array()
    array2=action2.as_array()
    if(array1.shape[0]!=array2.shape[0]):
        new_dim=min(array1.shape[0],array2.shape[0])
        array1=array1[:new_dim]
        array2=array2[:new_dim]
    new_array=np.concatenate((array1,array2),axis=1)
    return actions.Action(new_array,action1.name,
    	                    action1.cat,action1.person)
