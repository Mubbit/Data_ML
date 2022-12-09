#for new panrye, get doc2vec 
from data_preprocess_doc2vec import PreProcess
from gensim import Word2Vec
def getdoc2vec(s,stop_words_list,modelpath):
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

def getkobertvec(index,modelpath,pre_vec_path='case_to_vec.pt'):
  cv=torch.load(pre_vec_path)
  cvnpy=cv.detach().numpy()
  return cvnpy[index]