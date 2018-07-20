def indepen_measure(single_pred,all_pred):
    return [ sum(compare_pred(single_pred,pred_i,as_int=True))
                for pred_i in all_pred]
                    
def compare_pred(a,b,as_int=False):
    comp=[ a_i!=b_i for a_i,b_i in zip(a,b)]
    if(as_int):
        comp=np.array(comp).astype(int)
    return comp    