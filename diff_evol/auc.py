import sys
sys.path.append("..")
import exp,ens#learn

def auc_exp(common,binary,clf="LR"):
    datasets=ens.read_dataset(common,binary)
    dataset=[date_i.split() for date_i in datasets]
    dataset=list(zip(*dataset))
    train,test=dataset[0],dataset[1]
    print(len(dataset))
#    print(dataset[0].keys())

dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
auc_exp(paths["common"],paths["binary"],clf="LR")