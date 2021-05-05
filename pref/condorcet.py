import sys
sys.path.append("..")
from ens import Votes
import pref

def count_condor(in_path):
	votes=pref.read_pref(in_path)
	result=votes.voting(False)
	y_true,y_pred=result.as_labels()
	votes=pref.prepare_votes(votes)
	votes=[ pref.to_preference(vote_i) 
			for vote_i in votes]
	for i,win_i in enumerate(y_pred):
		is_condorcet_winner(win_i,votes[i])

def is_condorcet_winner(win_i,pref_i):
	
#	for ord_i in pref_i.order:
#		print(ord_i)
#		print("*******")
	raise Exception(win_i)


dataset="3DHOI"
count_condor("../s_SVC/%s" % dataset)