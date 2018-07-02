from metric import dtw_metric

class PairwiseDistance(object):
    def __init__(self,pairs):
        self.raw_pairs=pairs
   
    def __getitem__(self,index):	
        return self.raw_pairs[index]

    def has_pair(self,seq_a,seq_b):
        if(seq_a in self.raw_pairs):
            return seq_b in self.raw_pairs[seq_a]		
        return False

def make_pairwise_distance(actions):
    pairs_dict={ action_i.name:{}
                    for action_i in actions}
    pairs=PairwiseDistance(pairs_dict)
    for i,action_i in enumerate(actions):
        print("%i %s " % (i,action_i.name))
        for action_j in actions:
            if(pairs.has_pair(action_j.name,action_i.name)):
                pairs[action_i.name][action_j.name]=pairs[action_j.name][action_i.name]
            else:
            	pairs[action_i.name][action_j.name]=dtw_metric(action_i.img_seq,action_i.img_seq)
    return pairs

