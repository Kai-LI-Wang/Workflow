from sklearn.preprocessing import OneHotEncoder
import re 
import pandas as pd 
import datetime as dt 
import numpy as np 

class CheckData:
    def __init__(self):
        self.continuous, self.discrete,self.Additional,self.categorical_col = self.columns()

    def remove_ending_code(self,x):
        x = str(x)
        if re.findall("\.0",x ):
            x = re.sub("\.0","" ,x)
        if len(x) == 10:
            x = '0' + x 
        elif len(x) == 11:
            x = x 
        # print(x)
        return x    

    def String_to_datetime_sorted_apply(self, x):
        if isinstance(x, str):
            x = dt.datetime.strptime(x, "%Y-%m-%d")
            return x 
        elif isinstance(x, dt.datetime):
            return x 
        else:
            print("fail to convert into datetime. ")
        

    def String_to_datetime_sorted(self, df):
        if isinstance(df, pd.DataFrame): 
            # print("1: ",type(df['受理日期'].iloc[0]))
            df['受理日期'] = pd.to_datetime(df['受理日期'], format = '%Y-%m-%d')
            df.sort_values("受理日期", ascending = True, inplace = True)
            return df 
        elif isinstance(df, str):
            x = df 
            x = dt.datetime.strptime(x, "%Y-%m-%d")
            return x 
        
        elif isinstance(df, pd.Series): 
            df = pd.to_datetime(df, format = '%Y-%m-%d')
            return df 
        else:
            print("failed to convert into datetime. ")
            pass 
     
    def columns(self):       
        cat_cols = {}
        counter = 0 
        cols = pd.read_excel(r'C:\Users\bbkelly\Desktop\Kelly\BPI 效益資料\20211210模型重跑結果\reproduced_code_xlsm\factors.xlsx')
        for i in cols.columns: 
            temp = np.array(cols[i].astype(str))
            cat_cols[counter] = list(temp[temp!="nan"])
            if counter == 0: 
                continuous = cat_cols[counter]
            elif counter == 1:
                discrete = cat_cols[counter]
            elif counter == 2:
                Additional= cat_cols[counter]
            elif counter == 3:
                categorical_col= cat_cols[counter]
            counter += 1 
        return continuous, discrete,Additional,categorical_col
            
    def drop_redundant_col(self, df):
        if isinstance(df, pd.DataFrame):
            col_kept = [self.continuous + self.discrete + self.categorical_col]
            for col in df:
                if col not in col_kept:
                    df = df.drop(col, axis = 1)
            return df 
        else:
            print("Input argument must be DataFrame!")
            
    def Onehotencoder(self, X_train, X_test):
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(X_train)
        X_train_encode = pd.DataFrame(encoder.transform(X_train).toarray(), index = X_train.index)
        X_train_encode.columns = encoder.get_feature_names()
        X_test_encode = pd.DataFrame(encoder.transform(X_test).toarray(), index = X_test.index)
        X_test_encode.columns = encoder.get_feature_names()
        return X_train_encode, X_test_encode