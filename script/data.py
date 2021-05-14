import sys
sys.path.append("..")
import numpy as np
import exp,ens,files,learn,feats

def save_dataset(common,binary,out_path):
	datasets=ens.read_dataset(common,binary)
	files.make_dir(out_path)
	subdirs={desc_i:"%s/%s" % (out_path,desc_i) 
			for desc_i in ["features","votes"]}
	for path_i in subdirs.values():
		files.make_dir(path_i)
	for i,data_i in enumerate(datasets):
		feat_i="%s/%d" % (subdirs["features"],i)
		data_i.save(feat_i)
	results=learn.train_ens(datasets,clf="LR")
	for i,result_i in enumerate( results):
		votes_i="%s/%d" % (subdirs["votes"],i)
		text_i=list(result_i.y_pred.astype(str))
		lines=[]
		for line_j,name_j in zip(text_i,result_i.names):
			lines.append("%s#%s" % (line_j,name_j))
		files.save_txt(lines,votes_i)

dataset="3DHOI"
dir_path="../../ICSS"#%s" % dataset
paths=exp.basic_paths(dataset,dir_path,"dtw","ens/feats")
paths["common"].append("%s/%s/1D_CNN/feats" % (dir_path,dataset))
save_dataset(paths["common"],paths["binary"],'3DHOI_data')