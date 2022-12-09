#for new panrye, get doc2vec 
from case2vec_doc2vec import PreProcess
from gensim import Word2Vec
def getdoc2vec(s,modelpath,stop_words_list=['상 ','인 ','피 고 인']):
  #cleanse data
  p=PreProcess(s)
  tokens=p.run_infer(stop_words_list)

  #bring model
  model=Word2Vec.load(modelpath)

  #train model additionally
  model.train(tokens,total_examples=len(tokens))

  #get vectors
  vectors=[]
  for tok in tokens:
    vectors.append(model.infer_vector(tok))
  return vectors

def getkobertvec(s,index,modelpath='case_to_vec.pt'):
  
  if index!=-1:
    cv=torch.load(pre_vec_path)
    cvnpy=cv.detach().numpy()
  
    return cvnpy[index]
  else:
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    query_vec=model(s) 
    return query_vec
    