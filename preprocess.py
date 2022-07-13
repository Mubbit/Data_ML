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
                    #내부에는 Term, 그리고 uri 묶음이 있다
                    term.append(item[0].split('/')[-1])
                    print("The term of json object:",term[-1]) #Term과 나머지(uri 덩이)
                    uris.append(item[-1])#tuple of relevant uri 
                return term, uris #[단어], [(관련uri튜플)]
            except json.decoder.JSONDecodeError:
                print("String could not be converted to JSON")
                     
        print("failed to open file")
        return -1
        
    