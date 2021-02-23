import sys
sys.path.append("..")
import ens,person,acc as basic

def exp_person(common,binary,n,n_clf,clf="LR"):
	fun=person.total_person_selection
#	fun=basic.total_basic_selection
	all_clf=fun(common,binary,n,n_clf,clf)
	acc=[]
	for all_clf_i in all_clf:
		result=ens.ensemble(common,binary,clf,False,all_clf_i)[0]
		acc.append(result.get_acc())
	for acc_i,clf_set_i in zip(acc,all_clf):
		print(clf_set_i)
		print(acc_i)

if __name__ == "__main__":
    dataset="../../dtw_paper/MHAD"
    common="%s/common/MHAD_500" % dataset
    binary="%s/binary/1D_CNN/feats" %dataset
    exp_person(common,binary,1000,20,clf="LR")