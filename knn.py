import numpy as np
from metric import dtw_metric
import utils,pairs,dataset.instances
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix

class NNGraph(object):
    def __init__(self, distances,desc):
        self.distances=distances
        self.desc=desc

    def pred(self,name_i,k=10,admis=None):
        names=self.find_neighbors(name_i,k=k,admis=admis)
        neighbors=[ self.desc[name_i].cat  for name_i in names]
        count =Counter(neighbors)
        return count.most_common()[0][0]
    
    def find_neighbors(self,name_i,k=10,admis=None):
        names=admis if(admis) else self.desc.keys() 
        start=0 if(admis) else 1
        distance_i=[self.distances[name_i][name_j] 
                        for name_j in names]
        distance_i=np.array(distance_i)
        indexes=distance_i.argsort()[start:k]
        return [names[i] for i in indexes]

def knn(pairs_path,k=10):
    in_pairs=utils.read_object(pairs_path)    
    insts=pairs.get_descs(in_pairs)
    nn_graph=NNGraph(in_pairs,insts.as_dict())
    train,test=insts.split(None)
    y_true=test.cats()
    y_pred=[ nn_graph.pred(name_i,k,train.names()) 
                for name_i in test.names()]
    return y_pred,y_true

def check_prediction(pred_y,true_y):
    print(classification_report(pred_y,true_y,digits=4))
    print(confusion_matrix(pred_y,true_y))

def get_accuracy(matrix):
    return np.trace(matrix)/np.sum(matrix)

if __name__ == "__main__":
    pred_y,true_y=knn("mhad/pairs/corls",k=10)
    check_prediction(pred_y,true_y)