from pyemd import emd
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument,Word2Vec
from konlpy.tag import Kkma
import nltk
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os.path as osp

class PreProcess:
  def __init__(self,case):
    self.orig_df=case
  #1 df
  def set(self,columns=['datanumber','holding','summary','prny']):
    df=self.orig_df.copy()
    df['datanumber']=df['datanumber'].apply(str)
    df_joined=df[columns].agg(lambda x: ' '.join(x.values), axis=1).T

    self.df=pd.DataFrame(df_joined,columns=['text'])
    print('::::::::::df set',len(self.df))
  #2 re filter                    
  def filter(self):
    df=self.df.copy()
    pattern=re.compile('[^가-힣 ]')

    #re
    df_re=[]
    for row in list(df['text'].values):

      df_re.append(re.sub(pattern,' ',row))
    self.df['re']=df_re

    print(':::::::::::re done',len(df_re))
  #3 tokenize
  def tokenize(self,remove=False):
    if not osp.exists('tokenized.npy') or remove==True:
      kkma=Kkma()
      df=self.df['re'].copy()

      #tokens per doc
      tokens_per_doc=[]
      for test in tqdm(df):#per doc
        tokens_per_doc.append(pd.DataFrame(kkma.pos(test),columns=['tok','pos']))
      
      #all tokens

      all_tokens_df=pd.concat(tokens_per_doc)
      all_tokens_df.reset_index(inplace=True)
      all_tokens=list(all_tokens_df['tok'].values)

      #find frequent stopwords

      ko = nltk.Text(all_tokens, name='panrye_stopwords' )
      stop_words_list = []
      for m, n in ko.vocab().most_common(50):
          stop_words_list.append(m)

      #filter all docs of those stopwords
      tokens_per_doc_cleansed=[]
      for l in tokens_per_doc:#list of toks df per docs [tok,pos]
        tokens_per_doc_cleansed.append(list(l[~l['tok'].isin(stop_words_list)]['tok'].values))
      
      maxlen=max(map(len,tokens_per_doc_cleansed))
      tokens_per_doc_cleansed_padded=[tokens+[0]*(maxlen-len(tokens)) for tokens in tokens_per_doc_cleansed]
      #save
      #print([l for l in list(map(len,tokens_per_doc_cleansed_padded))])
      if len(tokens_per_doc_cleansed_padded)==1:
        np.save('tokenized.npy',np.array(tokens_per_doc_cleansed_padded).reshape((1,)))
        
      else:
        np.save('tokenized.npy',np.array(tokens_per_doc_cleansed_padded))
      np.save('stop_words_list.npy',np.array(stop_words_list))
      
      self.tokens=tokens_per_doc_cleansed #list of lists
    else:
      self.tokens=np.load('tokenized.npy')
      stop_words_list=np.load('tokenized.npy')
    return self.tokens, stop_words_list

  def tokenize_inference(self,stop_words_list):
    kkma=Kkma()
    df=self.df['re'].copy()

    #tokens per doc
    tokens_per_doc=[]
    for test in tqdm(df):#per doc
      tokens_per_doc.append(pd.DataFrame(kkma.pos(test),columns=['tok','pos']))

    #filter all docs of those stopwords
    tokens_per_doc_cleansed=[]
    for l in tokens_per_doc:#list of toks df per docs [tok,pos]
      print('l is type ',type(l))
      tokens_per_doc_cleansed.append(list(l[~l['tok'].isin(stop_words_list)]['tok'].values))
       
  def run(self):
    self.set()
    self.filter()
    return self.tokenize(remove=False)

  def run_infer(self,stop_words_list):
    self.set()
    self.filter()
    return self.tokenize_inference(stop_words_list)


from sklearn.metrics.pairwise import cosine_similarity

#랭킹 진행하고... 열 추가하기 
class Rank:
  def __init__(self,q,model):
    self.q=q 
    self.model=model
  def wmddistance(self,other):
    return self.model.wv.wmdistance(self.q,other)
  def cosdistance(self,other):
    return cosine_similarity(self.q,other)

def second_largest_index(li):
  if not type(li)==list:
    li=li.tolist()
  first=0
  second=0
  first=max(li) 
  first_index=li.index(first)
  if first_index<len(li)-1:
    second=max(li[:first_index]+li[first_index+1:])
    second_index=li.index(second)
  else:
    second=max(li[:-1])
    second_index=li.index(second)
  return second,second_index

def list_in_list_index(ll,l):
  for index,nl in enumerate(ll):
    if type(nl)==list and type(l)==list:
      if nl==l:
        return index 
    elif type(nl)==list:
      if nl.tolist()==l:
        return index 
    elif type(l)==list:
      if nl.tolist()==l:
        return index 
    else:
      if nl.tolist()==l.tolist():
        return index
  return -1
def ranking(qv,vectors,model,t='cos'): #find most similar item per item 
  ranker=Rank(qv,model)
  if t=='wmd':
    sim=list(map(ranker.wmddistance,vectors))
    max_sim_vec,max_sim_index=second_largest_index(sim)
  elif t=='cos':
    cos_adj=cosine_similarity(vectors,vectors)
    qv_index=list_in_list_index(vectors,qv)
    qv_sim=cos_adj[qv_index]
    max_sim_vec,max_sim_index=second_largest_index(qv_sim)
  return max_sim_vec,max_sim_index 
def recommend(cases,vectors,model,path_to_file,filename,t='cos'):
  recommendation=[]
  for i,vector in enumerate(vectors):
    max_rank_vec,max_rank_index=ranking(vector,vectors,model,t=t)
    next_item=cases.loc[max_rank_index,['datanumber']].values[0]
    recommendation.append(next_item)
  cases=pd.concat([cases,pd.DataFrame(recommendation,columns=['next_id'])],axis=1)
  cases.to_csv(osp.join(path_to_file,filename))

if __name__=='__main__':
    #case=pd.read_csv('case_list_with_textrank.csv')
    case=pd.read_csv('case_list_with_tfidf.csv')
    p=PreProcess(case)
    tokens,stop_words_list=p.run()
    
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
    common_texts=tokens
    
    remove=True
    
    if osp.exists('doc2vec.model') and remove==False:
      dmodel=Doc2Vec.load('D:\\PanryeAI\\doc2vec.model')
    else:
      dmodel = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
      dmodel.save('doc2vec.model')
    
    
    '''if osp.exists('word2vec.model') and remove==False:
      wmodel=Doc2Vec.load('D:\\PanryeAI\\word2vec.model')
    else:
      wmodel=Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
      wmodel.save('word2vec.model')'''
      
    if osp.exists('doc2vec.npy') and remove==False:
      doc2vec_np=np.load('doc2vec.npy')
    else:
      doc2vec=[]
      for tok in tokens:
          vector = dmodel.infer_vector(tok)
          doc2vec.append(vector)
      doc2vec_np=np.array(doc2vec) 
      
      np.save('doc2vec.npy',doc2vec_np)
      np.save('stopwords.npy',np.array(stop_words_list))
      
    
    
    #ranking
    
    recommend(case,doc2vec_np,dmodel,'D:\PanryeAI','case_list_w_doc2vec.csv',t='cos')
    #recommend(case,doc2vec_np,wmodel,'D:\PanryeAI','case_list_w_word2vec.csv',t='wmd')
    '''
    # 2차원 t-SNE 임베딩
    doc2vec=doc2vec_np.tolist()
    tsne_np = TSNE(n_components = 2).fit_transform(doc2vec)
    
    # numpy array -> DataFrame 변환
    tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
    
    #scatter
    plt.scatter(tsne_df['component 0'],tsne_df['component 1'],color='pink',label='panrye')
    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()
    plt.show()
    '''