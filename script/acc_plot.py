import numpy as np
import os,os.path
import matplotlib.pyplot as plt

def multi_plots(in_path,out_path):
	if(not os.path.isdir(out_path)):
		os.mkdir(out_path)
	paths={}
	for file_i in os.listdir(in_path):
		id_i=file_i.split('_')[0]
		if(not id_i in paths):
			paths[id_i]=[]
		paths[id_i].append("%s/%s" % (in_path,file_i))
	for id_i,path_i in paths.items():
		plot_i=make_multiplots(id_i,path_i)
		out_i="%s/%s.png" % (out_path,id_i)
		plot_i.savefig(out_i)

def make_multiplots(id_i,paths_i):
	plt.clf()
	plt.title(id_i)
	for path_j in paths_i:
		y=read_file(path_j)
		x=[ (i+1)*100 for i,acc_i in enumerate(y)]
		label_j=path_j.split('/')[-1]
		label_j=label_j.split(".")[0]
		label_j="_".join(label_j.split('_')[1:])
		plt.plot(x,y,label=label_j)
		plt.legend()
		print("%s,%s,%s" % (id_i,label_j,get_stats(y)))
	plt.ylabel('accuracy')
	plt.xlabel('number of features')
#	plt.show()
	return plt

def make_plots(in_path,out_path):
	if(not os.path.isdir(out_path)):
		os.mkdir(out_path)
	for file_i in os.listdir(in_path):
		in_i="%s/%s" % (in_path,file_i) 
		out_i="%s/%s.png" % (out_path,file_i.split('.')[0])
		acc=read_file(in_i)
		plot_i=show_plot(acc,show=False)
		plot_i.savefig(out_i)

def show_plot(acc,show=True):
	x=[ (i+1)*100 for i,acc_i in enumerate(acc)]
	plt.clf()
	plt.plot(x,acc)
	plt.ylabel('accuracy')
	plt.xlabel('number of features')
	if(show):
		plt.show()
	return plt

def to_file(acc,out_path):
	str_acc=str(acc)
	str_acc=str_acc.replace("[","")
	str_acc=str_acc.replace("]","")
	str_acc= "\n".join(str_acc.split(","))
	with open(out_path, 'w') as file:
		file.write(str_acc)

def read_file(in_path):
	acc=[]
	with open(in_path) as fp:
		lines = fp.readlines()
		for line_i in lines:
			acc.append(float(line_i))
	return acc

def get_stats(acc):
	stats_i=(np.mean(acc),np.std(acc),np.median(acc),np.amax(acc),np.amin(acc))
	str_i="%.4f,%.4f,%.4f,%.4f,%.4f"%stats_i
	return str_i
#to_file(acc,"raw/MHAD_stats.txt")
multi_plots("raw","plots")