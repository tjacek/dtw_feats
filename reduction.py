import exp,selection

def multi_exp(dtw,deep,binary,out_path,clf="LR"):
	fun=get_fun("selection")
	fun(dtw,None,out_path,clf=clf)
	fun(deep,None,out_path,clf=clf)
	fun(None,binary,out_path,clf=clf)
	fun(dtw,binary,out_path,clf=clf)
	fun(deep+dtw,binary,out_path,clf=clf)

def get_fun(gen_type):
	if(gen_type=="selection"):
#		raise Exception("OK")
		def fun(dtw,binary,out_path,clf):
			gen=selection.basic_selection
			return exp.single_exp(dtw,binary,out_path,clf=clf,fun=gen)
	else:
		def fun(dtw,binary,out_path,clf):
			return exp.single_exp(dtw,binary,out_path,clf=clf)
	return fun

deep=['../ICSS_exp/3DHOI/common/1D_CNN/feats']
binary='../ICSS_exp/3DHOI/ens/lstm/feats'
dtw=['../ICSS_exp/3DHOI/dtw/corl/person', '../ICSS_exp/3DHOI/dtw/max_z/person']

out_path="reduction/3DHOI_selection"
multi_exp(dtw,deep,binary,out_path)
exp.show_result(out_path,hard=False)
