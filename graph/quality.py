import numpy as np
import graph,utils,pairs
import itertools

class ClusterGraph(graph.Graph):
    def __init__(self, distances,desc):
#        train,test=desc.split()
        super(ClusterGraph, self).__init__( distances,desc)
    
    def quality(self):
        all_cats=self.by_cat()
        return [np.mean([ self.silhouette(name_i) 
                            for name_i in cat_i])
                                for cat_i in all_cats]  

    def silhouette(self,name_i):
        cat_i=self.desc[name_i].cat
        our_cat,other_cats=self.desc.split(lambda inst_i: inst_i.cat==cat_i)
        our_cat,other_cats=our_cat.names(),other_cats.names()
        a_i=self.avg_dist(name_i,our_cat)
        b_i=self.avg_dist(name_i,other_cats)
        return (b_i-a_i)/max([a_i,b_i])

    def avg_dist(self,name_i,names):
        dist=[self.distances[name_i][name_j] for name_j in names]
        return np.mean(dist)

    def by_cat(self):
        n_cats=self.desc.n_cats()
        return [self.get_cat(i) for i in range(n_cats)]

    def centroid(self,i):
        cat_names=self.desc.get_cat(i)
        dist_matrix=self.distance_matrix(cat_names,cat_names)
        total_dist=np.sum(dist_matrix,axis=0)
        index=np.argmax(total_dist)
        return cat_names[index]

def best_separation(in_path='mhad/deep_pairs',out_path='mhad/quality.txt'):
    paths=utils.bottom_files(in_path)
    dtw_graphs=[]
    for path_i in paths:
        print(path_i)
        dtw_pairs,insts=graph.read_dtw(path_i)
        dtw_graphs.append(ClusterGraph(dtw_pairs,insts))
    best_cls_ids=best_cls(dtw_graphs)
    insts=select_clusters(best_cls_ids,dtw_graphs)
    insts.to_txt(out_path)

def select_clusters(best_cls_ids,dtw_graphs):
    full_names=dtw_graphs[0].distances.keys()
    def get_feature(cls_id,feat_name):
        return dtw_graphs[cls_id].as_vector(feat_name,names=full_names)
    
    feats=[[  get_feature(cls_id,feat_name)
                for feat_name in dtw_graphs[cls_id].get_cat(cat_id)]
                    for cat_id,cls_id in enumerate(best_cls_ids)]

    feats=list( itertools.chain.from_iterable(feats))
    X=np.array(feats).T
    insts=pairs.get_descs(full_names)
    for i,name_i in enumerate(full_names):
        insts[name_i].data=X[i]
    return insts

def best_cls(dtw_graphs):
    quality=np.array([graph_i.quality()
                        for graph_i in dtw_graphs])
    print(quality)
    return np.argmin(quality,axis=0)