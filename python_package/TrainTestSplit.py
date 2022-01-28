from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd 
import datetime as dt 


class TTSplitting:
    
    def TrainTestSplitFunction(self, custom,TestSize=None, **group_data ):
        if custom == False: 
            TrainTestSplit = {}
            for key, value in group_data.items():
                X_train, X_test, y_train, y_test = train_test_split(group_data[key][0],group_data[key][1],\
                    test_size=TestSize,random_state=2, )
                TrainTestSplit[key] = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
            return TrainTestSplit 
        
        elif custom == True:
            TrainTestSplit = {}
            for key, value in group_data.items():
                df = pd.concat([group_data[key][0],group_data[key][1]], axis = 1)
                start = dt.datetime(2018,10,31)
                X_train = df[df['受理日期']<start].drop(["檢驗結果_重製",'受理日期','貨品分類號列'], axis = 1).astype(str)
                y_train = df[df['受理日期']<start]['檢驗結果_重製']
                X_test = df[df['受理日期']>=start].drop(["檢驗結果_重製",'受理日期','貨品分類號列'], axis = 1).astype(str)
                y_test = df[df['受理日期']>=start]['檢驗結果_重製']
                TrainTestSplit[key] = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
                if key == 'data':
                    try:
                        print('[TrainTest] X_train unique: ', X_train['生產國別'].unique())
                    except:
                        pass 
            return TrainTestSplit
