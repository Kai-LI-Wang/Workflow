import pyodbc
import pandas as pd 
import numpy as np 
import re 
from collections import Counter 

class retrieve_data_sql:
    def __init__(self,syntax_sql ):
        self.syntax_sql = syntax_sql
    def query_func(self):
        conn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};APP=MicrosoftR WindowsR Operating System;Trusted_Connection=Yes;SERVER=172.16.0.37')
        return pd.DataFrame(pd.read_sql(self.syntax_sql,conn))

class cleaning_data:
    def __init__(self, data):
        self.data = data
        self.data1 = self.remove_app_atom(self.data)
        self.data2 = self.remove_duplicate_code(self.data1)
        
    def remove_app_atom(self, data):
        re_df = data.copy()
        re_df.set_index(np.arange(re_df.shape[0]), inplace = True)
        result = []
        for i in range(re_df.shape[0]): 
            if re_df['案件註記'].iloc[i] is None: 
                if re_df['檢驗項目'].iloc[i] is not None:
                    x = re.findall(r"原子",re_df['檢驗項目'].iloc[i])
                    y = re.findall(r"外觀|標示",re_df['檢驗項目'].iloc[i])
                    if  len(x) > 0 and (re_df['國別中文名稱'].iloc[i]=='日本') :
                        result.append(i)
                    elif re.search(r"外觀|標示",re_df['檢驗項目'].iloc[i]):
                        result.append(i)
            elif re_df['案件註記'].iloc[i] is not None:
                result.append(i)
        df = re_df.drop(result, axis = 0).copy()
        df.index = range(df.shape[0]) # reset index or u will get wrong index 
        return  df 
        # re_df.fillna(value = "NaN", inplace = True)
        # test_ = re_df[re_df['檢驗項目']!='原子塵或放射能污染-原子塵或放射能污染']\
        #     .loc[re_df['檢驗項目']!='現場查核-中文標示及外觀']\
        #         .loc[re_df['案件註記']=="NaN"]

    def mapping(self, data):
        code = {}
        for i in range(data.shape[0]):
            t1 = str(data.loc[i, '簽審核准許可文件編號'])
            t2 = str(data.loc[i,'檢驗結果'])
            if t2 != 'None':
                if t1 not in code.keys():
                    code[t1] = t2
                elif code[t1] != 'N':
                    code[t1] =  t2 
        return code

    def find_duplicate(self, data, col):
        counter = dict(Counter(data[col]))
        code_repeat = []
        code_unique = []
        for i, j in counter.items():
            if j > 1: 
                code_repeat.append(i)
            else:
                code_unique.append(i)
        return code_repeat

    def check_duplicate_col(self, orginal):
        duplicate_columns = []
        for i,j in dict(Counter(orginal)).items():
            if j > 1:
                duplicate_columns.append(i)
        return duplicate_columns

    def remove_duplicate_code(self, df):
        code = self.mapping(df)
        df['簽審'] = df['簽審核准許可文件編號']
        df['簽審'] = df['簽審'].map(code)
        df = df.rename(columns={'簽審':'檢驗結果_重製'})
        code_repeat = self.find_duplicate(df, '簽審核准許可文件編號')
        df_no_dup = df.groupby(by = '簽審核准許可文件編號').last(numeric_only = False)
        df_no_dup['檢驗結果_重製'].fillna(value = "NaN", inplace = True)
        df_no_dup = df_no_dup[df_no_dup['檢驗結果_重製'] != 'NaN']
        df_no_dup = df_no_dup[df_no_dup['檢驗結果_重製']!= 'P']
        return df_no_dup 
        
    