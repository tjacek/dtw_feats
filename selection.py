import numpy as np,random
import ens,feats,learn,files,exp

def basic_selection(common_path,binary_path,clf="SVC"):
    datasets=ens.read_dataset(common_path,binary_path)
    s_datasets=dataset_selection(datasets)
    print(len(s_datasets))
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                for data_i in s_datasets]
    return ens.Votes(results)

def dataset_selection(datasets):
    acc=np.array([ validate_acc(data_i) for data_i in datasets])
    acc=np.array(acc)
    if(np.std(acc)==0):
    	return datasets
    acc= (acc-np.mean(acc))/np.std(acc)
    print(acc)
    s_datasets=[data_i for i,data_i in enumerate(datasets)
    			if(acc[i]>0)]
    return s_datasets

def validate_acc(data_i):
	train=data_i.split()[0]
	new_data=feats.Feats()
	for i,name_i in enumerate(train.keys()):
		new_name_i="%d_%d_%d" % (name_i.get_cat()+1,i%2,i)
		new_data[files.Name(new_name_i)]=train[name_i]
	result=learn.train_model(new_data,binary=False,clf_type="LR")
	return result.get_acc()

def selection_exp(dtw,deep,binary,out_path):
	fun=basic_selection
	exp.single_exp(None,binary,out_path,fun)
	exp.single_exp(dtw,binary,out_path,fun)
	for deep_i in deep:
		exp.single_exp(deep_i,binary,out_path,fun)
		exp.single_exp(dtw+[deep_i],binary,out_path,fun)

def random_selection(common_path,binary_path,n,n_cats,clf="LR"):
	votes=ens.make_votes(common_path,binary_path,clf,read=None)
	def helper(size):
		subset_i= random.sample(votes.results,size)
		votes_i=ens.Votes(subset_i)
		result_i=votes_i.voting(False)
		return result_i.get_acc()
	for k in range(2,n_cats):
		acc=[helper(k) for i in range(n)]
		print( max(acc))

if __name__ == "__main__":
	common_path="s_dtw"
	binary_path="../clean3/agum/ens/basic/feats"
#	votes=basic_selection(common_path,binary_path,clf="LR")
#	result=votes.voting(False)
#	result.report()
	random_selection(common_path,binary_path,1000,12)