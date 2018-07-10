import instances,utils
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import classification_report
from collections import Counter

class VotingEnsemble(object):
    def __init__(self,norm=False):
        self.norm=norm
    
    def __call__(self,deep_path):
        deep_paths=utils.bottom_files(deep_path)
        all_pred=[]
        y_true=None
        for feat_path_i in deep_paths:
            print(feat_path_i)	
            insts=instances.from_files(feat_path_i)
            train,test=instances.split_instances(insts)
            train,test=instances.to_dataset(train),instances.to_dataset(test)
            clf=LogisticRegression()
            clf = clf.fit(train.data, train.target)
            y_pred = clf.predict(test.data)
            if(y_true is None):
                y_true=test.target
            all_pred.append(y_pred)
        y_pred=self.vote(all_pred)
        print(classification_report(y_true, y_pred,digits=4))

        
    def vote(self,all_votes):
        all_votes=np.array(all_votes)
        y_pred=[]
        for vote_i in all_votes.T:
            count =Counter(vote_i)
            cat_i=count.most_common()[0][0]
            y_pred.append(cat_i)
        return y_pred 

