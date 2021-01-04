import learn,feats

def read_dataset(common_path,deep_path):
    if(not common_path):
        return feats.read(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=feats.read(deep_path)
    datasets=[common_data+ data_i 
                for data_i in deep_data]
    return datasets

def simple_exp(common_path,deep_path):
	datasets=read_dataset(common_path,deep_path)[0]
	acc=learn.train_model(datasets,binary=True,clf_type="LR",acc_only=True)
	print(acc)

common_path=["old2/agum/basic/feats","old2/simple/basic/feats"]
simple_exp(common_path,None)