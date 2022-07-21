import pandas as pd
import json
class ParseLemmJson():
    def __init__(self,fd):
        self.fd=fd
    def open(self):
            #표제어 파일 열기
        with open(self.fd,'r',encoding='UTF-8') as f:
            #json인식하기
            try:
                json_file=json.loads(f.read())
                print('json file opened')
                
                #json 내부 iterate하기 
                #keyword마다 iterate 되고 있다
                term=[]
                uris=[]
                for item in json_file.items(): 
                    #tuple인 item 내부에는 Term, 그리고 uri 묶음이 있다
                    term.append(item[0].split('/')[-1])
                    print("The term of json object:",term[-1], item[-1].keys()) #Term과 나머지(uri 덩이)
                    uris.append(item[-1])#tuple of relevant uri 
                return term, uris #[단어], [(관련uri튜플)]
            except json.decoder.JSONDecodeError:
                print("String could not be converted to JSON")
                     
        print("failed to open file")
        return -1
    def toPandas(self):
        term,uris=self.open()
        return pd.DataFrame(term,columns=['vocab'])

#test panrye
'''
dependency: bs4
'''
import socket
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
import requests
import re
class ParsePanrye:
    #check IP, set ID, connect
    def __init__(self) -> None:
        hostname=socket.gethostname()
        ipaddr=socket.gethostbyname(hostname)
        print('''
              class is initialized, 
              be sure your ip is authenticated by the api
              your ip: {}'''.format(ipaddr))
    def connect(self):
        self.myId=input('\ngive us your id').strip('\n')
        self.apilink='https://www.law.go.kr/DRF/lawSearch.do?OC={}&target=prec&type=XML'.format(self.myId)
        response=requests.get(self.apilink)
        soup = bs(response.text, 'xml')
        return soup
    def getPrecDetailLink(self):
        soup=self.connect()
        prec=soup.find_all('prec') # 리스트임
        precDetail=[]
        # for prec, get the link
        for p in prec:
            #find(pattern, str)
            alink='https://law.go.kr'+re.sub('<[^>]*>', '', str(p.find('판례상세링크')))
            alink=re.sub('amp;','',alink)
            alink=re.sub('&mobileYn=','',alink)
            precDetail.append(alink)
        return precDetail
    def parsePrecDetail(self):

        links=self.getPrecDetailLink() #우리가 구한 20개의 판례 링크들 
        self.detailLinks=[]
        #for prec, get connection
        # 변환을 해도 되는데 그냥 찾기로 진행함
        for l in links:
            response=requests.get(l) #works well 
            content=re.sub('\n*','\n',response.text)
            soup=bs(content,'html.parser')
            self.detailLinks+=soup.find('input',id='url').get('value').split('\n')
            
        for l in self.detailLinks:
            response=requests.get(l)
            content=re.sub('\r\n\r\n','',response.text)
            soup=bs(content,'html.parser')
            text=soup.get_text()
            print('mode 1 traversal',re.sub('\r\n','',text))
            
            #parse html and get
            ## 제목,요약, 본문 
            #title=soup.find(id='precNm')
            # #print(title)
        
    def test_tokenizePerP(tx):
        pass