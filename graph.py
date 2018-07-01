from collections import defaultdict
from metric import dtw_metric

class PairwiseDistance(object):
    def __init__(self):
        self.pairs={}#defaultdict(lambda:{})
   
    def __getitem__(self,index):
        if(not index in self.pairs):
            self.pairs[index]={}	
        return self.pairs[index]

    def has_pair(self,seq_a,seq_b):
    	if(seq_a in self.pairs):
            return seq_b in self.pairs[seq_a]		
    	return False

def make_pairwise_distance(actions):
    pairs=PairwiseDistance()
    for i,action_i in enumerate(actions):
        print("%i %s " % (i,action_i.name))
        for action_j in actions:
            if(pairs.has_pair(action_j.name,action_i.name)):
                pairs[action_i.name][action_j.name]=pairs[action_j.name][action_i.name]
            else:
            	pairs[action_i.name][action_j.name]=dtw_metric(action_i.img_seq,action_i.img_seq)
    return pairs

