import numpy as np
import utils,pairs

class Graph(object):
    def __init__(self, distances,desc):
        self.distances=distances
        self.desc=desc
    
    def get_cat(self,i):
        return self.desc.get_cat(i+1)

    def as_vector(self,name_i,names=None):
        if(not names):
            names=self.distances.keys()
            names.sort()
        dist_i=self.distances[name_i]
        return np.array([dist_i[name_j]   
                    for name_j in names])

def read_dtw(pairs_path):
    dtw_pairs=utils.read_object(pairs_path)    
    insts=pairs.get_descs(dtw_pairs)
    return dtw_pairs,insts