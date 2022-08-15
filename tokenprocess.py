import os
import json
#Tokenize text
from konlpy.tag import Kkma 
from konlpy.utils import pprint 
import pandas as pd
import numpy as np
#제nnn조 제n항
#규칙(0000. 0. 0. 대법원규칙 제 블라블라블라, 이하 '뫄뫄' 이라 한다)
#[별표]
#"뫄뫄뫄뫄 한다."

testdata={"test":' 당사자 사이에 소송비용을 일정 비율로 분담하도록 재판이 된 경우로서 민사소송법 제111조 제2항에 따라 소송비용액확정을 신청한 당사자에 대해서만 소송비용액을 확정할 경우 법원은 신청인으로부터 제출된 비용계산서에 기초하여 지출한 비용총액을 산정한 다음, 그 비용총액에 대하여 소송비용 부담재판의 분담비율에 따라 상대방이 부담할 소송비용액을 정하여 그 금액의 지급을 명하는 방법으로 소송비용액을 확정해야 한다. 한편 민사소송법 제109조 제1항은 "소송을 대리한 변호사에게 당사자가 지급하였거나 지급할 보수는 대법원규칙이 정하는 금액의 범위 안에서 소송비용으로 인정한다."라고 정하고 있고, 구 변호사보수의 소송비용 산입에 관한 규칙(2018. 3. 7. 대법원규칙 제2779호로 개정되기 전의 것, 이하 ‘구 보수규칙’이라 한다) 제3조 제1항은 "소송비용에 산입되는 변호사의 보수는 당사자가 보수계약에 의하여 지급한 또는 지급할 보수액의 범위 내에서 각 심급단위로 소송목적의 값에 따라 [별표]의 기준에 의하여 산정한다."라고 정하고 있다. 따라서 당사자 사이에 소송비용을 일정 비율로 분담하도록 재판이 된 경우로서 소송비용액확정신청을 한 신청인에게 피신청인이 상환해야 할 변호사 보수를 확정할 때에는 신청인이 변호사에게 보수계약에 따라 지급하거나 지급할 금액과 구 보수규칙에 따라 산정한 금액을 비교하여 그중 작은 금액을 소송비용으로 결정한 다음, 그에 대하여 소송비용 부담재판의 분담비율을 적용하여 계산해야 한다. '}

def processLaw(obj):
    if obj=="\"":
        return "[QT]"
    
class CleanseText:
    def __init__(self,yoji):
        
        self.yojiDict=yoji
        self.cleansed_text={}
    def cleanse(self):
        =None
        return text
    def tokenize(self):
        text=self.clease()
        pass
    def lemmatize(self):
        pass
    

cleanse=CleanseText(testdata)