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
	condor=[is_condorcet_winner(win_i,votes[i]) 
				for i,win_i in enumerate(y_pred)]
	print(condor)

def is_condorcet_winner(win_i,pref_i):
	n_cats=pref_i.order.shape[0]
	for cat_i in range(n_cats):
		if(cat_i!=win_i):
			if(not pref_i.pairwise_score(win_i,cat_i)):
				return False
	return True


dataset="3DHOI"
count_condor("../s_SVC/%s" % dataset)