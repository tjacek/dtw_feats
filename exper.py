import numpy as np
import ensemble.votes,ensemble.single_cls
import ensemble.stats,ensemble.multi_alg,plot

def cls_stats(in_path,basic_paths=None,multi_alg=True):
    if(multi_alg):
        multi_alg=ensemble.multi_alg.MultiAlgEnsemble()
        exp1=ensemble.stats.Experiment(multi_alg)
    else:
        exp1=ensemble.stats.Experiment()
    stats=exp1(in_path,basic_paths)
    ensemble.stats.show_stats(stats)
    desc=get_desc(basic_paths,multi_alg)
    vote_hist=stats['votes']
    plot.show_histogram(vote_hist,desc,cumsum=True)
    print("gini indec %f" % gini_index(vote_hist))

def get_desc(basic,multi_alg):
    alg="multi algorithm" if(multi_alg) else "deep"
    if(not basic):
        return alg
    if(type(basic)==str):
        feats=basic.split('/')[-1]
    else:
        names=[ basic_i.split('/')[-1] for basic_i in basic]
        names=[name_i.split(".")[0] for name_i in names]    
        feats="+".join(names)
    return alg+"\n"+feats 

def gini_index(hist):
    hist=np.sort(hist)
    cum_hist=np.cumsum(hist)
    n_bin=hist.shape[0]
    interv=1.0/(n_bin)
    lorenz=[(i+1)*interv for i in range(n_bin) ]
    print(lorenz)
    return np.sum(lorenz-cum_hist)/np.sum(lorenz)

basic_path=['mhad/simple/basic.txt','mhad/simple/max_z_feats.txt','mhad/simple/corls.txt']
#['mhad/simple/basic.txt','mhad/simple/max_z_feats.txt','mhad/simple/corls.txt']
#['mra/simple/basic.txt','mra/simple/max_z_feats.txt','mra/simple/corl_feats.txt']
adapt_path='mra/datasets'
 
cls_stats(in_path=adapt_path,basic_paths=None,multi_alg=False)
