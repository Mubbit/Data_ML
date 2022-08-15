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


# 시도 1: 
'''
메인에서 충분히 많은 판례의 리스트를 제공할 것이라고 착각했다. 
실제로 주어진 판례는 22년도 것 뿐으로 겨우 12건 정도 되는 것 같다. 
이를 기각하고 적어도 2000년부터 있었던 판례를 파싱해보려고 한다.
'''
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
        #print('connect activated',response.text)
        soup = bs(response.text, 'xml')
        return soup
    def getPrecDetailLink(self):
        soup=self.connect()
        prec=soup.find_all('prec') # 리스트임
        #print('soup.find_all activated',prec)
        precDetail=[]
        # for prec, get the link
        for p in prec:
            #find(pattern, str)
            alink='https://law.go.kr'+re.sub('<[^>]*>', '', str(p.find('판례상세링크')))
            alink=re.sub('amp;','',alink)
            alink=re.sub('&mobileYn=','',alink)
            precDetail.append(alink)
        #print("getPrecDetailLink activated",precDetail)
        return precDetail
    def parseMoreLinks(self):

        links=self.getPrecDetailLink() #우리가 구한 20개의 판례 링크들 
        self.detailLinks=[]
        #for prec, get connection
        # 변환을 해도 되는데 그냥 찾기로 진행함
        for l in links:
            #ids.append(re.findall('[0-9]{6}',l))
            response=requests.get(l) #works well 
            content=response.text
            soup=bs(content,'html.parser')
            self.detailLinks.append(soup.find('input',type='hidden',id='url').get('value'))
        #print("ParseMoreLinkes activated",self.detailLinks)
        #return ids #판례번호리스트
    def parsePrecDetail(self): 
        self.parseMoreLinks()
        match={}#판례번호:링크
        yoji={}#판례번호:판결요지
        for l in self.detailLinks:
            i=re.findall('[0-9]{6}',l)[0]
            match[i]=l#판례번호:링크
            res=requests.get(l)
            text=res.text
            soup=bs(text,'html.parser')
            interest=[t.get_text() for t in soup.find_all('p','pty4')]
            print(i,'\n',len(interest),'\n')
            yoji[i]=interest[1]
        return match, yoji
