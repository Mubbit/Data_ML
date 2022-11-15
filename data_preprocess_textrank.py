

# bring case list

# pre-process 판시사항, 판결요지, 판례내용 
# [가-힣만 남겨두기]
# 판결요지, 판례내용을 list of sentences로 만들기
# kkma tokenizer를 token list만 제공하는 형태로 post_fn을 만들어서 넘기기
# keyword와 keysentence를 구하기
# keyword와 keysentence 를 text 기준 위치를 구하는 함수를 만들기
# substring 비교식으로 해야할 것
from konlpy.tag import Kkma 
import pandas as pd
import numpy as np
import re

class PreProcess:#1개 이상의 판례를 대상으로 진행한다. 
    def __init__(self,case,patterns=re.compile('[^가-힣 ]')):
        self.ppsh=case.loc[:,'판시사항'].values #list of paragraphs
        self.pgyj=case.loc[:,'판결요지'].values #list of paragraphs
        self.prny=case.loc[:,'판례내용'].values #list of paragraphs
        self.filter_all(patterns)
    def filter(self,series,pattern=re.compile('[^가-힣 ]')):    #하나만 들어오더라도 []에 넣어주므로 동질적으로 진행 가능하다.
        new_series=[]
        for paragraph in series:
            new_series.append(re.sub(pattern,' ',paragraph))
        return new_series 
    def filter_all(self,patterns=re.compile('[^가-힣 ]')):
        if len(patterns)>1:
            print('not implemented for multiple pattern filtering')
        else:
            if isinstance(self.ppsh,list):
                self.filtered_df=pd.DataFrame({
				'판시사항':self.filter(self.ppsh),
				'판결요지':self.filter(self.pgyj),
				'판례내용':self.filter(self.prny)})
            else:
                self.filtered_df=pd.DataFrame({
				'판시사항':self.filter(self.ppsh),
				'판결요지':self.filter(self.pgyj),
				'판례내용':self.filter(self.prny)},index=[0])
				
class InferTextRank:
    def __init__(self,series):
        self.kkma=Kkma().noun
    def textrank(self,sent):
        return self.kkma(sent)
		
      
if __name__=='__main__':
    model=InferTextRank()
    case=pd.read_csv('case_list.csv')
    pre_process=PreProcess(case)
    #print(pre_process.filtered_df.head(5))
    for sent in pre_process.filtered_df:
        