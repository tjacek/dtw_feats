import sys
sys.path.append("..")
import os.path,re
import files,ens,learn

def full_exp(common,binary,out_path):
	if(type(binary)!=list):
		binary=[binary]
	simple_exp=SimpleExp()
	lines=["common,binary,n_clf,n_feats,accuracy,precision,recall,f1-score"]
	for binary_i in binary:
		lines+=simple_exp(common,binary_i)
	files.save_txt(lines,out_path)

def result_exp(prefix,result_dict):
    lines=[]
    for info_i,result_i in result_dict.items():
        line_i="%s,%s,%s" % (prefix,info_i,get_metrics(result_i))
        lines.append(line_i)
    files.save_txt(lines,out_path)

def exp_info(common_i,binary_i,result_i):
	desc_common=get_desc(common_i)
	desc_binary=get_desc(binary_i)
	return desc_common,desc_binary,get_metrics(result_i)

def get_metrics(result_i):
	acc_i= result_i.get_acc()
	metrics="%.4f,%.4f,%.4f" % result_i.metrics()[:3]
	return "%.4f,%s" % (acc_i,metrics)

def get_desc(common_path):
	if(common_path is None):
		return "-"
	if(type(common_path)==list):
		desc=[get_name(common_i) for common_i in common_path]
		return "-".join(desc)
	return get_name(common_path)

def get_name(path_i):
	return "_".join(path_i.split("/")[-2:])

def basic_paths(dataset,dir_path,common,binary,name="dtw"):
    paths={}
    paths["dir_path"]="%s/%s" % (dir_path,dataset)
    common="%s/%s" % (paths["dir_path"],common)
    paths["common"]=files.get_paths(common,name=name)
    if(binary):
        paths["binary"]="%s/%s" % (paths["dir_path"],binary)
    else:
        paths["binary"]=None
    return paths 

#def common_paths(common,binary):
#	common=files.top_files(common)
#	datasets=[ common_i.split('/')[-1].split("_")[0] 
#				for common_i in common]
#	binary=[ binary % data_i  for data_i in datasets]
#	return list(zip(common,binary))