import re
import pandas as pd
from konlpy.tag import Kkma 
from konlpy.utils import pprint
import numpy as np

patterns=[
    re.compile('[가-힣]+법\s제[0-9]+조\s제[0-9]+항\s제[0-9]호',flags=re.I|re.X),
    re.compile('[가-힣]+법\s제[0-9]+조\s제[0-9]호',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+령\s제[0-9]+조\s제[0-9]+항\s제[0-9]호',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+규칙\s제[0-9]+조\s제[0-9]+항\s제[0-9]호',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+령\s제[0-9]+조\s제[0-9]+항',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+규칙\s제[0-9]+조\s제[0-9]+항',flags=re.I|re.X),
	re.compile('[가-힣]+법\s제[0-9]+조\s제[0-9]+항',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+령\s제[0-9]+조',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+규칙\s제[0-9]+조',flags=re.I|re.X),
	re.compile('[가-힣]+법\s제[0-9]+조',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+령',flags=re.I|re.X),
	re.compile('[가-힣]+법\s[가-힣]+규칙',flags=re.I|re.X),
 ]
datepattern= re.compile('([0-9]+).\s([0-9]+).\s([0-9]+)',flags=re.I|re.X)
class Cleanse:
    global patterns

    #문장 하나에 대해서
    def remove_s(self,pattern,t): 
        #cleanse law
        got=re.findall(pattern,t)
        for s in got:
            alt=re.sub('\s','_',s)
            alt=re.sub('[\·]','_',alt)
            #print(alt)
            t=re.sub(s,alt,t)
        return t
    #문단 하나에 대해서 법률 단어는 연결시켜놓기 
    def group_law(self,data):
        for p in patterns:
            data=self.remove_s(p,data)
        
        data=re.sub(',','$',data)#이제부터 쉼표는$
        fin=re.sub(datepattern,r'\1\2\3',data)#cleanse date
        return fin
    
    #문단 자르기 
    def cut_meaning(self,data):
        splitted=re.split('\[[0-9]+\]',data)[1:]
        return splitted
    
    def tokenize(self,data):
        kkma=Kkma()
        word_dict=pd.DataFrame(columns=['vocab','pos','sent_id','eojeol_id'])
        highlightable=pd.DataFrame(columns=['highlight','sent_id','eojeol_id'])
        for j, sent in enumerate(data.split('.')):#j 는 n번째 문장이라는 뜻
            for k, word in enumerate(sent.split(' ')):#k는 n번째 어절이라는 뜻 
                if bool(re.match('[\s\n]+',word)) or word=="":
                    continue
                passage_hs=kkma.pos(word) #list in list
                highlight=pd.DataFrame([[word,j,k]],columns=['highlight','sent_id','eojeol_id'])
                highlightable=pd.concat([highlightable,highlight],axis=0)
                for hs in passage_hs: #어절 내부의 위치는 기록하지 않는다. 어절은 통째로 뺄 것이기 때문이다. 
                
                    position=[j,k]+list(hs)
                    current=np.reshape(np.array(position),(1,-1))
                    current=pd.DataFrame(current,columns=['sent_id','eojeol_id','vocab','pos'])
                    word_dict=pd.concat((word_dict,current),axis=0)
        
        return word_dict, highlightable
 
class Cleansed_iterated:
    def __init__(self,df):
        self.df=df
        self.cleanser=Cleanse() 
    def __iter__(self): 
        self.passage_id=0
        self.passage=None
        self.pid=None
        self.word_dict=[]
        self.highlightable=[]
        return self
    def __next__(self):
        self.passage=self.df.iloc[self.passage_id]
        self.pid=self.passage['판례일련번호']
        data=self.cleanser.group_law(self.passage['판결요지'])
        splitted_data=self.cleanser.cut_meaning(data)
        for i,s in enumerate(splitted_data):
            w,h=self.cleanser.tokenize(s)#요지마다 어근어미 구분되어 토큰화된 pd df와 어절으로 토큰화된 df를 받음 
            self.word_dict.append(w)
            self.highlightable.append(h)
            #판례번호, 요점 번호, 요점당 word_dict, 요점당 highlight할 수 있는 어절    
        self.passage_id+=1#return 전에 한 번 올라가서, 실상 1부터 시작하는 구조
        return self


#사용 예시
'''a=Cleansed_iterated(case_list)
k=iter(a)
for i,m in enumerate(k):
    print(m.word_dict)
    if i>3:
        break'''
			