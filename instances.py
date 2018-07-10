import numpy as np
import seqs.io
import feats,plot

class Instance(object):
    def __init__(self,data,cat,person,name):
        self.data = data
        self.cat=cat
        self.person=person
        self.name=name

    def  __str__(self):
        feats=[ str(feat_i) for feat_i in list(self.data)]
        feats=",".join(feats)
        return "%s#%s#%s#%s" % (feats,self.cat,self.person,self.name)

def to_dataset(instances):
    X,y=[],[]
    for inst_i in instances:
        X.append(np.array(inst_i.data))
        y.append(inst_i.cat)	
    X=np.array(X)
    return X,y

if __name__ == "__main__":
    action_path="mra/feats/nn19"
    read_actions=seqs.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_actions(action_path)
    feat_extractor=feats.GlobalFeatures()
    inst=[feat_extractor(action_i) for action_i in actions]
    X,y=to_dataset(inst)
    plot.plot_embedding(X,y,highlist=[20])