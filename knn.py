import numpy as np
from metric import dtw_metric
import utils,pairs,dataset.instances
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix

class Graph(object):
    def __init__(self, distances,desc):
        self.distances=distances
        self.desc=desc

    def as_vector(self,name_i,names=None):
        if(not names):
            names=self.distances.keys()
            names.sort()
        dist_i=self.distances[name_i]
        return np.array([dist_i[name_j]   
                    for name_j in names])

class ClusterGraph(Graph):
    def __init__(self, distances,desc):
        train,test=desc.split()
        super(ClusterGraph, self).__init__( distances,train)

    def quality(self):
        cats=self.by_cat()
        cat_sizes=[ float(len(cat_i)) for cat_i in cats]
        size=float(len(self.desc))
        a,b=[],[]
        for i,cat_i in enumerate(cats):
            b_i=[np.sum(self.distance_matrix(cat_i,cat_j))
                    for j,cat_j in enumerate(cats)
                        if(i!=j)]
            a_i=np.sum(self.distance_matrix(cat_i,cat_i))/cat_sizes[i]
            b_i=np.sum(b_i)/(size-cat_sizes[i])
            a.append(a_i)
            b.append(b_i)            
        return silhouette(a,b)   
    
    def by_cat(self):
        n_cats=self.desc.n_cats()
        return [self.desc.get_cat(i+1) for i in range(n_cats)]

    def centroid(self,i):
        print(i)
        cat_names=self.desc.get_cat(i)
        dist_matrix=self.distance_matrix(cat_names,cat_names)
        total_dist=np.sum(dist_matrix,axis=0)
        index=np.argmax(total_dist)
        return cat_names[index]

    def distance_matrix(self,a_names,b_names):
        dist=[[ self.distances[name_i][name_j]
                for name_j in a_names]
                    for name_i in b_names]
        return np.array(dist)

def silhouette(a,b):
    return [  (b_i-a_i)/max([a_i,b_i])  
                for a_i,b_i in zip(a,b)]

def best_separation(in_path='mhad/deep_pairs',out_path='mhad/quality.txt'):
    paths=utils.bottom_files(in_path)
    dtw_graphs=[]
    for path_i in paths:
        print(path_i)
        dtw_pairs,insts=read_dtw(path_i)
        dtw_graphs.append(ClusterGraph(dtw_pairs,insts))
    best_cls_ids=best_cls(dtw_graphs)
    names=dtw_graphs[0].distances.keys()
    feat_names=[ dtw_graphs[cls_i].centroid(i+1) 
                    for i,cls_i in enumerate(best_cls_ids)]
    feats=[ dtw_graphs[cls_i].as_vector(feat_names[i],names) 
                for i,cls_i in enumerate(best_cls_ids)]
    X=np.array(feats).T
    insts=pairs.get_descs(names)
    for i,inst_i in enumerate(insts):
        print(X[i].shape)
        inst_i.data=X[i]
        print(inst_i.data.shape)
    dataset.instances.to_txt(out_path,insts)

def best_cls(dtw_graphs):
    quality=np.array([graph_i.quality()
                        for graph_i in dtw_graphs])
    return np.argmin(quality,axis=0)

class NNGraph(Graph):
    def __init__(self, distances,desc):
        super(NNGraph, self).__init__( distances,desc.as_dict())

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
    dtw_pairs,insts=read_dtw(pairs_path)
    nn_graph=NNGraph(dtw_pairs,insts)
    train,test=insts.split(None)
    y_true=test.cats()
    y_pred=[ nn_graph.pred(name_i,k,train.names()) 
                for name_i in test.names()]
    return y_pred,y_true

def read_dtw(pairs_path):
    dtw_pairs=utils.read_object(pairs_path)    
    insts=pairs.get_descs(dtw_pairs)
    return dtw_pairs,insts

def check_prediction(pred_y,true_y):
    print(classification_report(pred_y,true_y,digits=4))
    print(confusion_matrix(pred_y,true_y))

def get_accuracy(matrix):
    return np.trace(matrix)/np.sum(matrix)

if __name__ == "__main__":
    best_separation(in_path='mhad/deep_pairs')
#    dtw_pairs,insts=read_dtw("mhad/pairs/max_z")
#    cluster_graph=ClusterGraph(dtw_pairs,insts)
#    a,b=cluster_graph.averages()
#    print(silhouette(a,b))
#    pred_y,true_y=knn("mhad/pairs/max_z",k=10)
#    check_prediction(pred_y,true_y)