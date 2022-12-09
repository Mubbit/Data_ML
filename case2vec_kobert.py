import pandas as pd 
import numpy as np 
import re
import torch
#try 1
from KoBERT.kobert import get_tokenizer
from KoBERT.kobert_tokenizer import KoBERTTokenizer
import torch
from transformers import BertModel
model = BertModel.from_pretrained('skt/kobert-base-v1')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

#case prny를 가공해서 kobert 만들기
def preprocess(passage):
    stopwords='[○\(\)\[\]\{\}\$\【\】\,,외,선고,법,원고탈퇴,피,]'
    passed=re.sub(stopwords,'',passage)
    passed=re.sub('피 고 인','피고인',passed)
    passed=re.sub('호 소 인','호소인',passed)
    passed=re.sub('피  인','피인',passed)
    passed=re.sub('호 인','호인',passed)
    passed=re.sub('상 ',' ',passed)
    passed=re.sub('겸 ',' ',passed)
    passed=re.sub(' 인',' ',passed)
    passed=re.sub('[ ]+',' ',passed)
    passed=re.sub('[가-힣]+\.','*',passed)
    passed=re.sub('\.','',passed)
    return passed
def bert_input_fy(passage):
    return passage.split('*')


case=pd.read_csv('case_list.csv')
case.head(5)

prny=case.copy()['prny'].apply(preprocess)

bert_input=[]
for p in prny:
    bert_input+=bert_input_fy(p)
    

inputs=tokenizer.batch_encode_plus(bert_input,pad_to_max_length=True)

out = model(input_ids = torch.tensor(inputs['input_ids']),
              attention_mask = torch.tensor(inputs['attention_mask']))


torch.save(out.pooler_output, '../dataset/case_nm_tensor.pt')