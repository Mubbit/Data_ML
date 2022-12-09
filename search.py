import sys
from run_doc2vec_kobert_infer import getdoc2vec,getkobertvec
import numpy as np
import torch
from sklearn import cosine_similarity
import pandas as pd 

def compare(vec,t='doc2vec',n=10):
    if t=='doc2vec':
        other_vecs=np.load('doc2vec.npy')
    else:
        cv=torch.load('case_to_vec.pt')
        other_vecs=cv.detach().numpy()
    def cosine_sim(other_vec):
        return cosine_similarity(vec,other_vec)
    result=np.array(list(map(cosine_sim,other_vecs)))
    max_arg=np.argsort(result)
    return max_arg[:n]
    
    
if __name__=='__main__':
    query,t=sys.argv[0],sys.argv[1]
    
    if t=='doc2vec':
        vec=getdoc2vec(query,'doc2vec.model')
        
        max_args=compare(vec,t='doc2vec')
        
    else:
        vec=getkobertvec(query,-1,'case_nm_vec.pt')
        
        max_args=compare(vec,t='doc2vec')
    
    
    cases=pd.read_csv('case_list.csv')
    
    cases.loc[max_args,'datanumber'].to_csv('search_result.csv')