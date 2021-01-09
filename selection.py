import numpy as np
import ens,feats,learn,files

def basic_selection(common_path,binary_path,clf="SVC"):
    datasets=ens.read_dataset(common_path,binary_path)
    acc=np.array([ validate_acc(data_i) for data_i in datasets])
    acc=np.array(acc)
    acc= (acc-np.mean(acc))/np.std(acc)
    s_datasets=[data_i for i,data_i in enumerate(datasets)
    			if(acc[i]>0)]
    print(acc)
    print(len(s_datasets))
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                for data_i in s_datasets]
    return ens.Votes(results)

def validate_acc(data_i):
	train=data_i.split()[0]
	new_data=feats.Feats()
	for i,name_i in enumerate(train.keys()):
		new_name_i="%d_%d_%d" % (name_i.get_cat()+1,i%2,i)
		new_data[files.Name(new_name_i)]=train[name_i]
	result=learn.train_model(new_data,binary=False,clf_type="LR")
	return result.get_acc()

deep=['../ICSS_exp/MSR/common/stats/feats']
binary='../ICSS_exp/MSR/ens/lstm/feats'
dtw=['../ICSS_exp/MSR/dtw/corl/dtw', '../ICSS_exp/MSR/dtw/max_z/dtw']
result=basic_selection(dtw+deep,binary,clf="LR")
print(result.voting().get_acc())