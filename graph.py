import numpy as np
from metric import dtw_metric
import utils,actions.io

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

class PairwiseDistance(object):
    def __init__(self,pairs):
        self.raw_pairs=pairs
   
    def __getitem__(self,index):	
        return self.raw_pairs[str(index)]

    def has_pair(self,seq_a,seq_b):
        seq_a=str(seq_a)
        seq_b=str(seq_b)
        if(seq_a in self.raw_pairs):
            return seq_b in self.raw_pairs[seq_a]		
        return False

    def as_matrix(self,names=None):
        if(names is None):
            name=self.raw_pairs.keys()
            names.sort()
        distance=[[ self.raw_pairs[name_i][name_j]
                        for name_i in names]
                            for name_j in names]
        return np.array(distance)

def make_nn_graph(pairs_path,action_path,k=100):
    pairs=utils.read_object(pairs_path)    
    vertex_neighbors={ name_i:find_neighbors(distance_i,k)
                        for name_i,distance_i in pairs.items()}
    
    read_actions=actions.io.ActionReader(as_dict=True)
    seqs=read_actions(action_path)
    return NNGraph(vertex_neighbors,seqs,k)

def make_pairwise_distance(actions):
    pairs_dict={ action_i.name:{}
                    for action_i in actions}
    pairs=PairwiseDistance(pairs_dict)
    for i,action_i in enumerate(actions):
        print("%i %s " % (i,action_i.name))
        for action_j in actions:
            if(pairs.has_pair(action_j,action_i)):
                pairs[action_i][action_j]=pairs[action_j][action_i]
            else:
            	pairs[action_i][action_j]=dtw_metric(action_i.img_seq,action_j.img_seq)
    return pairs

def find_neighbors(distances,k):
    names=distances.keys()
    values=np.array(distances.values())
    indexes=values.argsort()[1:k]
    return [names[i] for i in indexes]