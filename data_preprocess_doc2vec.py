from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from konlpy.tag import Kkma
import nltk
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
  def tokenize(self):
    kkma=Kkma()
    df=self.df['re'].copy()

    #tokens per doc
    tokens_per_doc=[]
    for test in tqdm(df):#per doc
      tokens_per_doc.append(pd.DataFrame(kkma.pos(test),columns=['tok','pos']))
    
    print('::::::::::::tokens per doc length',len(tokens_per_doc))
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
    
    self.tokens=tokens_per_doc_cleansed
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
    return self.tokenize()

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
    return self.model.wmdistance(self.q,other)
  def cosdistance(self,other):
    return cosine_similarity(self.q,other)

def second_largest_index(li):
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
def ranking(qv,vectors,model,t='wmd'): #find most similar item per item 
  ranker=Rank(qv,model)
  if t=='wmd':
    sim=list(map(ranker.wmddistance,vectors))
    max_sim_vec,max_sim_index=second_largest_index(sim)
  elif t=='cos':
    sim=list(map(ranker.cosdistance,vectors))
    max_sim_vec,max_sim_index=second_largest_index(sim)
  return max_sim_vec,max_sim_index 
def recommend(cases,model,path_to_file):
  recommendation=[]
  for i,vector in enumerate(cases['vector'].values):
    max_rank_vec,max_rank_index=ranking(case,cases,model)
    next_item=cases.loc[i,['datanumber']]
    recommendation.append(next_item)
  cases=pd.concat([cases,pd.DataFrame(recommendation,columns=['nextid'])],axis=1)
  cases.to_csv(osp.join(path_to_file,'case_list_with_textrank_with_nextid.csv'))
    
if __name__=='__main__':
    case=pd.read_csv('case_list_with_textrank.csv')
    p=PreProcess(case)
    tokens,stop_words_list=p.run()
    
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
    if osp.exists('doc2vec.model'):
      model=Doc2Vec.load('D:\\PanryeAI\\doc2vec.model')
    else:
      model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
      model.save('doc2vec.model')
      
      
    if osp.exists('doc2vec.npy'):
      doc2vec_np=np.load('doc2vec.npy')
    else:
      doc2vec=[]
      for tok in tokens:
          vector = model.infer_vector(tok)
          doc2vec.append(vector)
      doc2vec_np=np.array(doc2vec) 
      
      np.save('doc2vec.npy',doc2vec_np)
      np.save('stopwords.npy',np.array(stop_words_list))
      
    
    
    #ranking
    case_w_vec=pd.concat([case,pd.DataFrame(list(map(str,doc2vec_np)),columns=['vector'])],axis=1)
    recommend(case,model,'D:\PanryeAI')
    
    # 2차원 t-SNE 임베딩
    tsne_np = TSNE(n_components = 2).fit_transform(doc2vec)
    
    # numpy array -> DataFrame 변환
    tsne_df = pd.DataFrame(tsne_np, columns = ['component 0', 'component 1'])
    
    #scatter
    plt.scatter(tsne_df['component 0'],tsne_df['component 1'],color='pink',label='panrye')
    plt.xlabel('component 0')
    plt.ylabel('component 1')
    plt.legend()
    plt.show()