def indepen_measure(y_true,all_pred):
    all_pred=np.array( all_pred).T
    def indepen_helper(i,pred_i,pred_ij):
        if(y_true[i]==pred_ij):
            print(i)
            return 0
        bool_array=(pred_i==pred_ij)
        indepen_ij=sum(bool_array.astype(int))
        return indepen_ij-1

    indep=[ [indepen_helper(i,pred_i,pred_ij)
                for pred_ij in pred_i]
                    for i,pred_i in enumerate(all_pred)]
    indep=np.matrix(indep) 
    print(indep.shape)
    return indep