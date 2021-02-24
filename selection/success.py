import numpy as np
import ens

def success_selection(common_path,binary_path,clf="LR"):
    votes=ens.make_votes(common_path,binary_path,clf="LR")
    n_cats=len(votes)
    acc_matrix=[[ result_i.cat_acc(cat_i) 
    				for result_i in votes.results]
    					for cat_i in range(n_cats)]
    acc_matrix=np.array(acc_matrix)
    best=np.argmax(acc_matrix,axis=0)
