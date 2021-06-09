import sys
sys.path.append("..")
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import ens,learn,exp

class Corl(object):
    def __init__(self,all_votes):
        self.all_votes=ens.Votes(all_votes)	
        self.d=[ result_i.true_one_hot() 
                  for result_i in all_votes]
    
    def __call__(self,weights):
        weights=weights/np.sum(weights)
        results=self.all_votes.results
        C=self.corl(results,self.d)
        n_clf=len(self.all_votes)
        loss=0
        for i in range(n_clf):
            for j in range(n_clf):
                loss+=weights[i]*weights[j] * C[i,j] 	
        return 1.0*loss

def corl(results,d):
    n_clf=len(results)
    C=np.zeros((n_clf,n_clf))
    for i in range(n_clf):
        for j in range(n_clf):
            f_i=results[i].y_pred
            f_j=results[j].y_pred
            c_ij= (f_i-d)*(f_j-d)
            C[i,j]=np.mean(c_ij)
    return C

def visualize_corl(paths):
    datasets=ens.read_dataset(paths["common"],paths["binary"]) 
    results=[learn.train_model(data_i) for data_i in datasets]
    d=[ result_i.true_one_hot() for result_i in results]
    C=corl(results,d)
    heat_map(C)

def heat_map(matrix,title="3DHOI_corl"):
    sn.set(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(6,6))
    sn.heatmap(matrix,cmap='Greys',
        annot=True,annot_kws={"size": 6}, fmt='g')
    if(title):
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    dataset="3DHOI"
    dir_path="../../ICSS"#%s" % dataset
    paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
    paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
    visualize_corl(paths)