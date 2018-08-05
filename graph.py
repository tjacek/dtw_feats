import numpy as np
from metric import dtw_metric
import utils,seqs.io

class NNGraph(object):
    def __init__(self, vertex_neighbors,insts,k):
        self.vertex_neighbors=vertex_neighbors
        self.insts=insts
        self.k=k 
    
    def get_neighbors(self,name_i,n=10):
        names=self.vertex_neighbors[name_i]
        return [self.get_cat(name_j) for name_j in names]

    def all_cats(self):
        return [inst_i.cat
                for inst_i in self.insts.values()]

    def get_cat(self,name_i):
        return self.insts[name_i].cat

    def names(self):
        return self.vertex_neighbors.keys()
        
def make_nn_graph(pairs_path,action_path,k=100):
    pairs=utils.read_object(pairs_path)    
    vertex_neighbors={ name_i:find_neighbors(distance_i,k)
                        for name_i,distance_i in pairs.items()}
    insts=pairs.get_descs(pairs,as_dict=False)
    return NNGraph(vertex_neighbors,insts,k)

def find_neighbors(distances,k):
    names=distances.keys()
    values=np.array(distances.values())
    indexes=values.argsort()[1:k]
    return [names[i] for i in indexes]