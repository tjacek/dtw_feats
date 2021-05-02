import sys
sys.path.append("..")
import os.path,re
import files,ens,learn

class SimpleExp(object):
	def __init__(self,fun=None,clf="LR"):
		if(not fun):
			fun=ens.ensemble
		self.fun=fun
		self.clf=clf

	def __call__(self,common,binary):
		if(type(common)!=list):
			common=[common]
		common=[None]+common
		lines=[]
		for common_i in common:
			result_i=self.fun(common_i,binary,clf=self.clf)[0]
			desc_i=clean_info(common_i,binary,result_i)
			lines.append(desc_i)#"%s,%s,%s" % desc_i)
		return lines

def full_exp(common,binary,out_path):
	if(type(binary)!=list):
		binary=[binary]
	simple_exp=SimpleExp()
	lines=["common,binary,n_clf,n_feats,accuracy,precision,recall,f1-score"]
	for binary_i in binary:
		lines+=simple_exp(common,binary_i)
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

def clean_info(common_i,binary_i,result_i):
	desc=exp_info(common_i,binary_i,result_i)
	n_feats=get_n_feats(desc[0])
	desc_common="-" if(n_feats=="-") else "dtw"
	desc_binary=desc[1].replace("_feats","")
	n_clf=result_i.n_cats()
	clean=(desc_common,desc_binary,n_clf,n_feats,desc[2])
	return "%s,%s,%d,%s,%s" % clean

def get_n_feats(common_desc):
    if(common_desc=="-"):
    	return "-"
    n_feats=re.findall('\d+',common_desc)
    if(len(n_feats)==0):
    	return "full"
    return str(n_feats[0])

if __name__ == "__main__":
	dataset="MHAD"
	dir_path="../../dtw_paper/%s/common" % dataset
	common1=files.get_paths("%s/feats" % dir_path)
	common2="%s/%s_300" % (dir_path,dataset) 
	common3="%s/%s_350" % (dir_path,dataset) 
	common=[common1,common2,common3]
	binary1="../../dtw_paper/%s/sim/feats" % dataset
	binary2="../../dtw_paper/%s/stats/feats" % dataset
	binary3="../../dtw_paper/%s/1D_CNN/feats" % dataset
	binary=[binary1,binary2,binary3]
	full_exp(common,binary,"MHAD.txt")