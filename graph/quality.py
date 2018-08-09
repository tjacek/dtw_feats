import numpy as np
import graph,utils,pairs

class ClusterGraph(graph.Graph):
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
        return [self.get_cat(i) for i in range(n_cats)]

    def centroid(self,i):
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
        dtw_pairs,insts=graph.read_dtw(path_i)
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
        inst_i.data=X[i]
    insts.to_txt(out_path)

def best_cls(dtw_graphs):
    quality=np.array([graph_i.quality()
                        for graph_i in dtw_graphs])
    print(quality)
    return np.argmin(quality,axis=0)