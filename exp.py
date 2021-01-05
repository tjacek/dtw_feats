import files,ens,learn
#import pickle

def ens_exp(common_path,binary_path,out_path):
	files.make_dir(out_path)
	for clf_i in ["LR","SVC"]:
		for binary_i in [True,False]:
			result=ens.ensemble(common_path,binary_path,binary=binary_i,clf=clf_i)
			out_i="%s/%s_%s" % (out_path,clf_i,str(binary_i))
			result.save(out_i)

def show_result(in_path):
	print("clf,Hard voting,accuracy,precision,recall,fscore")
	for path_i in files.top_files(in_path):
		result_i=learn.read(path_i)
		metrics_i=result_i.metrics()
		metrics_i= [result_i.get_acc()] +list(metrics_i[:3])
		metrics_i="%s,%s,%s,%s" % tuple(metrics_i)
		prefix_i=path_i.split('/')[-1].replace("_",",")
		line_i="%s,%s" % (prefix_i,metrics_i)
		print(line_i)

common_path="good/ae_basic"
binary_path="good/ens"
#ens_exp(common_path,binary_path,"results")
show_result("results")