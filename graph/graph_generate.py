import dgl 
import pandas as pd
import re

class GraphGenerator:
    '''This is a generator for a one big graph '''
    def __init__(self):
        self.nodes=pd.DataFrame(columns=['pr_id','feature']) #판례 id, doc2vec
        self.edges=pd.DataFrame(columns=['src','dst']) 
    
    def add_node(self,df):
        '''do it additively starting from an empty list'''
        pass
    
    def parse_visited_panrye(self,df):
        revised_df=df[['판례일련번호','참조판례']].copy()
        marker='\[[0-9]\]'
        
        for row in df:
            q=row.copy()
            if q['참조판례'].str.contains(marker):
                q_list=re.split(re.compile(marker),q['참조판례'].str)
                #search for the panrye id of all q elements
                for k in q_list:
                    #find
                    pass
            elif q['참조판례']=='-':
                continue
            else:
                #search for the single panrye
            
        return revised_df
    def add_edge(self,df):
        '''do it additively starting from an empty list'''
        pass


if __name__=="__main__":
    pass