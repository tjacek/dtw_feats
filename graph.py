import numpy as np
from metric import dtw_metric
import utils,seqs.io

class NNGraph(object):
    def __init__(self, vertex_neighbors,actions,k):
        self.vertex_neighbors=vertex_neighbors
        self.actions=actions
        self.k=k 
    
    def get_neighbors(self,name_i,n=10):
        names=self.vertex_neighbors[name_i]
        return [self.get_cat(name_j) for name_j in names]

    def all_cats(self):
        names=self.actions.keys()
        return [self.get_cat(name_i)
                for name_i in names]

    def get_cat(self,name_i):
        return self.actions[name_i].cat

    def names(self):
        return self.vertex_neighbors.keys()
def make_nn_graph(pairs_path,action_path,k=100):
    pairs=utils.read_object(pairs_path)    
    vertex_neighbors={ name_i:find_neighbors(distance_i,k)
                        for name_i,distance_i in pairs.items()}
    
    read_actions=actions.io.ActionReader(as_dict=True)
    seqs=read_actions(action_path)
    return NNGraph(vertex_neighbors,seqs,k)

def find_neighbors(distances,k):
    names=distances.keys()
    values=np.array(distances.values())
    indexes=values.argsort()[1:k]
    return [names[i] for i in indexes]