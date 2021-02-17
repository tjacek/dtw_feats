import numpy as np,random
import feats,files,learn

def rename_exp(in_path,k=10,threshold=None):
	data=feats.read(in_path)[0]
	group=files.by_cat(data.keys())
	acc=[]
	for i in range(k):
		print(i)
		rename=random_split(group)
		new_data=rename_dataset(data,rename)
		result=learn.train_model(new_data,binary=False,clf_type="LR")
		acc_i=result.get_acc()
		if(threshold and acc_i>threshold):
			print(acc_i)
			return rename
		acc.append(acc_i)
	print(np.mean(acc),np.std(acc),np.amax(acc),np.amin(acc))

def rename_dataset(dataset,rename):
	new_data=feats.Feats()
	for name_i in dataset.keys():
		new_name=rename[name_i]
		new_data[new_name]=dataset[name_i]
	return new_data

def random_split(group):
	rename={}
	for class_i in group.values():
		random.shuffle(class_i)
		for i,name_i in enumerate(class_i):
			if( (i%2)==0):
				new_name="%d_0_%d" % (name_i.get_cat()+1,i)
			else:
				new_name="%d_1_%d" % (name_i.get_cat()+1,i)
			rename[name_i]=files.Name(new_name)
	return rename

rename=rename_exp("s_dtw",k=1000,threshold=0.7)
#print(names)