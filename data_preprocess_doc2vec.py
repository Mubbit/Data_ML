import tqdm
import pandas as pd
import numpy as np
import re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
class PreProcess:
  def __init__(self,case):
    self.orig_df=case
  #1 df
  def set(self,columns=['사건명','판시사항','판결요지','판례내용']):
    df=self.orig_df.copy()
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

    all_tokens=list(pd.concat(tokens_per_doc).reset_index(inplace=True)['tok'].values)

    #find frequent stopwords

    ko = nltk.Text(all_tokens.values, name='panrye_stopwords' )
    stop_words_list = []
    for m, n in ko.vocab().most_common(50):
        stop_words_list.append(m)

    #filter all docs of those stopwords
    tokens_per_doc_cleansed=[]
    for l in tokens_per_doc:#list of toks df per docs [tok,pos]
      print('l is type ',type(l))
      tokens_per_doc_cleansed.append(list(l[~l['tok'].any().isin(stop_words_list)]['tok'].values))
    
    self.tokens=tokens_per_doc_cleansed
    return self.tokens, stop_words_list
   
  def run(self):
    self.set()
    self.filter()
    return self.tokenize()


if __name__=='__main__':
	p=PreProcess(case)
	tokens,stop_words_list=p.run()
 
	documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tokens)]
	model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)
 
	np.save('doc2vec_panrye.npy',model.wv)
