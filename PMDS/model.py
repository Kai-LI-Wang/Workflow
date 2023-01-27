'''
### Reference 
@knn_impute(): https://gist.github.com/YohanObadia/b310793cd22a4427faaadd9c381a5850

'''
from sklearn.metrics import auc
from itertools import chain
from collections import Counter 
import pickle 
import re 
import time 
import os 
import pyodbc
import pandas as pd 
import numpy as np 
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest,chi2, f_classif, SelectFromModel
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC #lasso
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB,BernoulliNB
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt 
import plotly.express as px 
import warnings
warnings.filterwarnings("ignore")
# from sklearn.base import ClassifierMixin
# pd.options.display.float_format = "{:,.5f}".format 
import xgboost as xgb 


global IsGridSearch
global Impute_method
global Criteria 
global StartYear 

IsGridSearch = True   
Impute_method = 'knn'
Criteria = 'precision'
StartYear = 2021

class retrieve_data:
    def connect(self):
        conn = pyodbc.connect('''
                DRIVER={ODBC Driver 17 for SQL Server};
                APP=MicrosoftR WindowsR Operating System;
                Trusted_Connection=Yes;SERVER=172.16.0.37''')
        return conn 

    ##合併PMDS與非登_KAI
    def get_data(self ):
        self.conn = self.connect()
        start = time.time()
        cursor = self.conn.cursor()
        exe = cursor.execute("""select * from ##業者稽查紀錄_kai""")
        業者稽查紀錄 = pd.DataFrame(np.array(exe.fetchall()), columns = [i[0] for i in cursor.description])
        exe = cursor.execute("""select * from ##變數串聯_kai""")
        合併PMDS與非登_KAI = pd.DataFrame(np.array(exe.fetchall()), columns = [i[0] for i in cursor.description])
        
        if 業者稽查紀錄.shape[0] == 0:
            raise ValueError("Failed to extract data from ##業者稽查紀錄_kai.")
        elif 合併PMDS與非登_KAI.shape[0]==0:
            raise ValueError("Failed to extract data from ##變數串聯_kai.")
        
        合併PMDS與非登_KAI['稽查結果_'] = 合併PMDS與非登_KAI['稽查結果'].apply(lambda x: 1  if x == '限期改善' else 0 )
        合併_ = 合併PMDS與非登_KAI[['市招名稱','稽查結果_']].groupby('市招名稱').max()
        合併PMDS與非登_KAI['稽查結果_改'] = 合併PMDS與非登_KAI['市招名稱'].map(dict(zip(合併_.index.tolist(),合併_['稽查結果_'].tolist())))
        合併PMDS與非登_KAI.drop('稽查結果_', axis=1, inplace = True)
        end = time.time()
        print("time spent = ",end - start)
        return 業者稽查紀錄, 合併PMDS與非登_KAI

    # def close_db(self):
    #     cursor = self.conn.cursor()
    #     cursor.execute('''drop table if exists ##業者稽查紀錄_kai''')
    #     cursor.execute('''drop table if exists ##合併PMDS與非登_KAI''')
    #     cursor.close()



class test:
    def 稽查結果_測試(self, X):
        a = X[['業者代碼','稽查結果','稽查結果_改']].drop_duplicates()['業者代碼'].value_counts()
        b = X[X['業者代碼'].isin(a[a>1].index.tolist())][['業者代碼','稽查結果','稽查結果_改']]
        return b
    
    def 確認同業者同時有缺失值與非缺失值(self):
        pass 
    
    def check_miss(self, func):
        X = func()
        a = X.isnull().sum()
        if a.sum() != 0:
            raise "補值失敗, {} 還有缺失值.".format(a[a>0].index.tolist())
        else:
            "Imputation is successfull."
            
class DataPreprocessing(test):
    
    def __init__(self,測試資料起始年份 = 2021, 
                 Impute_method = 'knn'):
        
        self.測試資料起始年份 = 測試資料起始年份
        # self.全部因子 = list(chain(*[i for i in self.get_type_columns(check=False)]))
        self.Impute_method = Impute_method
        
        
    def get_type_columns(self, data_ = None,check = False):
        因子類別 = pd.read_excel('./因子分類.xlsx')
        因子類別 = 因子類別[因子類別['features'].isin(data_.columns.tolist())]
        
        key_cols = 因子類別[因子類別['type']=='key']['features'].tolist()
        con_cols = 因子類別[因子類別['type']=='continuous']['features'].tolist()
        cat_cols = 因子類別[因子類別['type']=='categorical']['features'].tolist()
        dis_cols = 因子類別[因子類別['type']=='discrete']['features'].tolist()
        target = 因子類別[因子類別['type']=='target']['features'].tolist()
        columns = {'key':[], 'cotinuous':[], 'categorical':[], 'discrete': [], 'target':[]}
        all_ = key_cols + con_cols + cat_cols + dis_cols + target 
        self.全部因子 = all_.copy()
        
        if check:
            for col in self._check_missing_value(data_[all_]):
                if col in key_cols: 
                    columns['key']+= [col] 
                elif col in con_cols: 
                    columns['cotinuous']+= [col] 
                elif col in cat_cols: 
                    columns['categorical']+= [col]
                elif col in dis_cols:
                    columns['discrete']+= [col]
                elif col in columns['target']:
                    columns['target'] += [col]
                    
            return columns['key'], columns['cotinuous'],columns['categorical'], columns['discrete'],columns['target']
        else:
            return key_cols,con_cols,cat_cols,dis_cols , target
    
    def log_transoform(self, X: pd.Series):
        return np.log(X+1)
    
    def scaler(self):
        pass 
    
    def onehot(self, X):
        one = OneHotEncoder(handle_unknown='ignore')
        X_encoded = pd.DataFrame(one.fit_transform(X).toarray(), columns=one.get_feature_names(X.columns))
        self.one = one 
        return X_encoded
    
    def chi_square(self, X, y):
        chi2, p, dof, ex = chi2_contingency(pd.crosstab(X, y))
        #chi2(data[columns], y)
        return p

    def anova(self,X, y ):

        if not isinstance(X, pd.DataFrame):
            pvalue = f_classif(X.values.reshape(-1,1),y.values.reshape(-1,1))[1][0]
        else:
            pvalue = f_classif(X.values, y.values.reshape(-1,1))[1][0]
        return pvalue

    def convert_datatype(self, data_, columns: list, target_type = 'continuous'):
        
        if target_type == 'continuous':
            if isinstance(columns, list):
                for i in columns:
                    try:
                        data_[i] = data_[i].apply(lambda x: np.nan if x == '' else x)
                        data_[i] = data_[i].astype(int)
                    except:
                        print("failed to convert {} into integer from {}".format(i, data_[i].dtype))
                        try:
                            data_[i] = data_[i].astype(np.float64)
                        except:
                            print("failed to convert {} into np.float64 from {}".format(i, data_[i].dtype))
            else:
                print("'columns' argument must be a list.")
        return data_ 
        
    def train_test_split(self, X , year = 2021):
        X_test = X[X['稽查年份'] >= year].drop(['稽查年份', '稽查結果_改'], axis=1)
        y_test = X[X['稽查年份'] >= year]['稽查結果_改']
        # X_val = data[data['稽查年份']=='2021'].drop(drop_cols, axis = 1)
        # y_val = data[data['稽查年份']=='2020']['稽查結果_改']
        X_train = X[X['稽查年份'] < year].drop(['稽查年份', '稽查結果_改'], axis=1)
        y_train = X[X['稽查年份'] < year]['稽查結果_改']
        X_cat = self.onehot(X[self.類別因子])
        X_non_cat = X[[i for i in X.columns if i not in self.類別因子]]
        X_encoded = pd.concat([X_non_cat,X_cat ],axis=1)
        X_train_encoded = X_encoded[X_encoded['稽查年份'] < year].drop(['稽查年份', '稽查結果_改'], axis=1)
        X_test_encoded = X_encoded[X_encoded['稽查年份'] >= year].drop(['稽查年份', '稽查結果_改'], axis=1)
        
        return X_train, y_train, X_test, y_test,X_train_encoded,X_test_encoded
        
    def _check_missing_value(self, data_):
        # missing_stat = data.isnull().sum()
        self.missing_table = pd.DataFrame(data_.isnull().sum())
        self.missing_table['percentage'] = self.missing_table[0] / data_.shape[0]
        self.missing_table.rename(columns={0:'NumOfNull', 'percentage':'佔總資料筆數'},inplace = True)
        return self.missing_table[self.missing_table['NumOfNull']>0]['NumOfNull'].sort_values().index.tolist()
    
    def _label_convert(self, X , columns):
        label = LabelEncoder()
        
        for i in columns:
            NonNull_idx = X[X[i].notnull()].index 
            # Null_idx = X[X[i].isnull()].index 
            X.loc[NonNull_idx, i] = label.fit_transform(X.loc[NonNull_idx, i])
        return X
            
    def label_inverse(self,data_ , columns):
        pass 
    
    def _imputation(self, X,columns = None,cat_cols = None, method = 'knn') -> pd.DataFrame:
        '''
        X: array-like or pd.DataFrame
        method: specify which method applied for dealing with missing value.
                -> 'knn' : utilize sklearn.preprocessing.KNNImputer 
                -> 'mean': pd.DataFrame.fillna(X[column].mean())
                -> 'most_frequent': pd.DataFrame.fillna(df['Label'].value_counts().index[0])
        columns: if method is 'mean' or 'most_frequent', need to specify columns 
        cat_cols: if method is 'knn', need to specify categorical columns
        '''
        start = time.time()
        if method == 'knn':
            X_cat_label = self._label_convert(X,cat_cols)
            X[cat_cols] = X_cat_label[cat_cols]
            imputer = KNNImputer(n_neighbors=5, copy = False)
            X = pd.DataFrame(imputer.fit_transform(X), columns = X.columns)
        elif method == 'mean':
            for col in columns:
                X[col] = X[col].fillan(X[col].mean())
        elif method == 'most_frequent':
            for col in columns:
                X[col] = X[col].fillan(X[col].value_counts().index[0])
        end = time.time()
        print('Time spent for imputation is %s'% (end - start) )
        return X
    
    # @test.check_miss
    def _fill_miss_value(self, data_,target = '稽查結果_改')->pd.DataFrame:
        print(data_[target].value_counts())
        self.key因子, self.連續因子,self.類別因子, self.離散因子, self.目標因子 = self.get_type_columns(data_ = data_,check=False)
        # self.全部因子 = list(chain(*[連續因子,類別因子, 離散因子]))
        缺失連續因子,缺失類別因子,缺失離散因子 = self.get_type_columns(data_ = data_,check=True)[1:4]
        缺失連續及離散因子 = 缺失連續因子+缺失離散因子
        
        全部缺失因子 = 缺失連續因子+缺失類別因子+缺失離散因子
        # df = pd.DataFrame(imputer.fit_transform(data[self.all_col]), columns = self.all_col)
        drop_cols = []
        for col in 全部缺失因子:
            遺失值佔比 = self.missing_table[self.missing_table.index == col]['佔總資料筆數'].values[0]
            if col in 缺失連續及離散因子:
                pvalue = self.anova(X = data_[~data_[col].isnull()][col], y = data_[data_[col].notnull()]['稽查結果_改'])
                if pvalue > 0.05 or 遺失值佔比 > 0.5:
                    data_.drop(col, axis = 1, inplace = True)
                    drop_cols.append(col)
                    if col in self.全部因子:
                        self.全部因子.remove(col) # include key, index and target columns
                    if col in self.連續因子:
                        self.連續因子.remove(col)
                    elif col in self.離散因子:
                        self.離散因子.remove(col)
                    
            elif col in 缺失類別因子:
                pvalue = self.chi_square(X = data_[data_[col].notnull()][col], y = data_[data_[col].notnull()]['稽查結果_改'])
                if pvalue > 0.05 or 遺失值佔比 > 0.5:
                    data_.drop(col, axis = 1, inplace = True)
                    drop_cols.append(col)
                    self.類別因子.remove(col)
                    if col in self.全部因子:
                        self.全部因子.remove(col) # include key, index and target columns
          
        # cat_cols = [i for i in 類別因子 if i not in drop_cols]

        # variable_name = [i for i in 連續因子+類別因子+離散因子+[target] if i not in drop_cols]
        variable_name = self.連續因子 + self.類別因子 + self.離散因子 + [target]
        df = data_[variable_name].copy()
        df_impute = self._imputation(X = df, cat_cols = self.類別因子)
        self.因子 = df_impute.columns.tolist()
        df_impute = pd.concat([df_impute, data_[self.key因子]], axis=1)
        return df_impute



class BuildModel:
    
    def __init__(self):
        pass 
    
    def confusion_matrix(self, y_test, y_pre = None):
        if type(y_pre)==type(None):
            a = np.r_[self.y_pre, y_test].reshape(2,-1)
        else:
            a = np.r_[y_pre, y_test].reshape(2,-1)
        b = a.T
        TP = sum(b[:,0] * b[:,1] == 1)
        FN = sum((b[:,1] == 1 )& (b[:,0] == 0 ))
        FP = sum((b[:,1] == 0 )& (b[:,0] == 1 ))
        TN = sum((b[:,1] == 0 )& (b[:,0] == 0 ))
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        recall = TP/(TP+FN)
        precision = TP/(FP+TP)
        accuracy = (TP+TN)/sum([TP,FP,TN,FN])
        F1 = 2*precision*recall/(precision+recall)
        return [TP,FP,FN, TN,recall, precision,accuracy, F1, TPR, FPR]
    
    def performance(self, X_test, y_test):
        r = []
        for i in range(0,11):
            pre = self.predict(X_test, threshold=i/10)
            temp = self.confusion_matrix(y_test, y_pre = pre)
            temp.insert(0,i/10)
            r.append(temp)
        thre_table =  pd.DataFrame(
            r, 
            columns = ['threshold','TP','FP','FN', 'TN','recall', 'precision','accuracy', 'F1', 'TPR','FPR']
        )
        return thre_table
    
    def ModelSelect(self, 
                    X_train,
                    y_train, 
                    criteria = 'precision',
                    n_splits=5,
                    model_selected=None):
        '''
        criteria: 'f1', 'recall', 'precision', 'accuracy'
        n_splits: number of kfold split 
        model_selected: if it is none, one of models including 'xgb', 'rf', 'Logis', 'BerNB will be chosen 
                        based on 'criteria'
        '''
        print("Model seletion start.")
        start = time.time()
        kfold = KFold(n_splits=n_splits, shuffle = True)
        models = self.model()
        _Goup = []
        for idx, (train_idx , test_idx) in enumerate(kfold.split(X_train)):
            train_x = X_train.iloc[train_idx, :]
            train_y = y_train.iloc[train_idx]
            test_x = X_train.iloc[test_idx, :]
            test_y = y_train.iloc[test_idx]
            
            for name, mod in models.items():
                # print(name)
                self.CLF = mod
                self.fit(train_x,train_y)
                
                self.predict(train_x)
                mat = self.confusion_matrix(train_y)
                mat1_ = [name,'train',idx] + mat 
                _Goup.append(mat1_)
                
                self.predict(test_x)
                mat = self.confusion_matrix(test_y)
                mat2_ = [name,'validation',idx] + mat 
                _Goup.append(mat2_)
        
        result = pd.DataFrame(_Goup, 
                columns = ['model', 'data','Group','TP','FP','FN', 'TN','recall', 'precision','accuracy', 'f1', 'TPR', 'FPR']
            ).groupby(['model','data']).mean()
        self.result = result
        result = result[np.array(list(result.index))[:,1]=='validation']
        
        
        # result[np.array(list(result.index))[:,0]==model_selected][['recall', 'precision']]
        if model_selected == None:
            model_selected = result.index[np.argmax(result[criteria])][0]
        
        self.CLF = models[model_selected]       
        end = time.time()
        print("Total time spent for model selection is %d seconds."% (end - start))    
        return result      
    
    def model(self):
        models = {
            'xgb': XGBClassifier(objective = 'binary:logistic',learning_rate = 0.01),
            'rf': RandomForestClassifier(),
            'logis': LogisticRegression(max_iter = 400),
            'BerNB': BernoulliNB()
            # 'SVC': SVC(probability = True)
        }
        return models
        
    def fit(self, X_train, y_train):
        self.CLF.fit(X_train, y_train)
        return self 
    
    def predict(self, X_test, threshold=0.5):
        self.y_pre_prob = self.CLF.predict_proba(X_test)[:,1] # np.array([合格, 不合格])
        self.y_pre = (self.y_pre_prob >= threshold).astype(int) 
        # return self.y_pre

class HyperParameterTunning(BuildModel):
    
    def __init__(self,criteria, model):
        self.ModelName = re.search("\w+",str(model)).group() 
        self.criteria = criteria
        self.model = model 

    def GridOverfitCheck(self, X, y, grid):
        y_pre = grid.predict(X)
        model = BuildModel()
        model.CLF = grid
        self.thres_table = model.performance(X, y)
        return self.confusion_matrix(y_test = y, y_pre = y_pre)
    
    def Parameters(self, ModelName):
        parameters = {
            
        'RandomForestClassifier':[{
            'n_estimators': [50, 100, 300],
            'criterion': ['gini'],
            'min_samples_split':[5,20,50],
            # 'min_impurity_decrease':[0.0001, 0.001,0.01],
            # 'max_samples':['sqrt', 'log2'],
            'ccp_alpha':[0.001, 0.01]}],
        
        'XGBClassifier':[{
            'n_estimators':[200,500],
            'eta':[0.01, 0.001],
            }]
            
        'BernoulliNB':[{
            'alpha':[0.5,1]
            }]
        
        }
        return parameters[ModelName]
        
    
    def GridSearch(self, X_train, y_train, X_test, y_test):
        print('GridSearch is called.')
        start = time.time()
        r = []
        parameters = self.Parameters(self.ModelName)
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=parameters,
            scoring = self.criteria,
            return_train_score=True)
        
        grid.fit(X_train,y_train)
        self.grid = grid 
        
        for X, y in zip([X_train,X_test ],[y_train, y_test]):
            r.append(self.GridOverfitCheck(X, y , grid.best_estimator_))
        self.r = r 
        print(r)
        end = time.time()
        print("Total time spent for grid search is %d" % (end - start))
        return grid.best_params_, grid.best_score_ 

class FeatureSelection(BuildModel, DataPreprocessing):
    
    def __init__(self,連續因子, 離散因子,類別因子, criteria = 'f1'):
        self.連續因子 = 連續因子
        self.離散因子 = 離散因子
        self.類別因子 = 類別因子
        self.criteria = criteria
        self.CLF = None 
    
    def Lasso(self, X_train, y_train):
        '''
        X_train: after encoding trainning data set 
        '''
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=False)
        model.fit(X_train, y_train)
        self._lasso_coef = pd.DataFrame(lsvc.coef_.reshape(-1,1), index = X_train.columns.tolist(),columns=['lasso'])

        X_train = pd.DataFrame(model.transform(X_train), columns =  X_train.columns[model.get_support()])
        # print("lsvc.coef_ shape = ", lsvc.coef_.shape)
        self.coef_ = lsvc.coef_
        return X_train
    
    def RandomForestImportance(self, X_train, y_train):
        RF_CLF = self.model()['rf']
        select = SelectFromModel(RF_CLF)
        select.fit(X_train, y_train)
        _importance = select.estimator_.feature_importances_
        rf_col_selected = select.get_support()
        return _importance, rf_col_selected 
    
    def SelectMethod(self,X_train,y_train,X_test,y_test, method = None):
        '''
        method: 'lasso', 'anova_chi' , 'rf'
        '''
        start = time.time()
        anova, ch2, temp, Prediction, models, Performacne = [], [], [], {}, {}, {}
        
        result = pd.DataFrame(index = X_train.columns.tolist(), columns = ['anova_chi'])
        
        X_train_lasso = self.Lasso(X_train, y_train)
        X_test_lasso = X_test[[i for i in X_test.columns if i in X_train_lasso.columns]]
        
        for col in X_train.columns:
            re_col = re.search('[\u4e00-\u9fff|\d|A-Z]+',col ).group()
            if re_col in self.類別因子:
                ch2.append(col)
                pvalue = self.chi_square(X_train[col], y_train)
                result.loc[col,'anova_chi'] = pvalue
            elif re_col in self.連續因子 + self.離散因子:
                anova.append(col)
                # print("3:")
                
        if len(anova) != 0:
            pvalue = self.anova(X_train[anova], y_train)
            result.loc[anova, 'anova_chi'] = pvalue
        
        col_stat = result[result['anova_chi'] <= 0.05].index.tolist()
        X_train_stat = X_train[col_stat]
        X_test_stat = X_test[col_stat]
        result['lasso'] = self._lasso_coef.loc[result.index.tolist(),'lasso']
        
        _importance, rf_col_selected = self.RandomForestImportance(X_train,y_train)
        
        result['rf'] = _importance
        X_train_rf = X_train[X_train.columns[rf_col_selected]]
        X_test_rf = X_test[X_test.columns[rf_col_selected]]
        print("""The model used to evaluate which feature 
              select method is appropriate is '{}'""".format(
                                re.search("\w+\(",str(self.CLF)).group().replace('(',"")
                                ))
        
        # check model performance after anova and chi-square feature selection 
        for n, m in zip([ 'anova_chi','lasso', 'rf'],
                        [X_train_stat,X_train_lasso,X_train_rf ]):
            self.CLF.fit(m, y_train)
            X_test_cp = X_test[[i for i in X_test if i in m.columns]]
            pre = self.CLF.predict_proba(X_test_cp)
            Prediction[n] = pre 
            pre = (pre >= 0.5).astype(int)
            temp.append(self.confusion_matrix(y_test = y_test, y_pre = pre))
            
            # Performacne[n] = self.performance(X_test_cp, y_test)
            models[n] = self.CLF
            
        index = ['TP','FP','FN', 'TN','recall', 'precision','accuracy', 'f1', 'TPR', 'FPR']
        result = pd.concat([result,pd.DataFrame(np.array(temp).T,
                   columns = ['anova_chi','lasso', 'rf'], 
                   index = index)], 
                    axis = 0)
        
        result['statisc_drop'],result['Lasso_drop'],result['RF_drop'] = 'Yes','Yes','Yes'
        result.loc[col_stat, 'statisc_drop'] = 'No'
        result.loc[X_train_lasso.columns.tolist(), 'Lasso_drop'] = 'No'
        result.loc[X_train_rf.columns.tolist(), 'RF_drop'] = 'No'
        result.loc[index, ['statisc_drop','Lasso_drop','RF_drop']] = np.nan 
        
        self.result = result
        methods = {'lasso':[X_train_lasso,X_test_lasso], 
                   'anova_chi':[X_train_stat,X_test_stat],
                   'rf':[X_train_rf,X_test_rf]}
        
        idx = result.columns[np.argmax(result.loc[self.criteria,: ])]
        METHOD = methods[idx] if method == None else methods[method]
        self.prediction = Prediction[idx] if method == None else Prediction[method]
        self.MODEL = models[idx] if method == None else models[method]
        
        Prediction,end = None,time.time() 
        print("Time spent for feature selection is %d seconds."% (end - start))
        return METHOD


class GraphPlot:
    def __init__(self):
        plt.figure(figsize=(10,15))
    
    def FeatureImportance(self,data, X, y):
        fig = px.bar(data ,x = X ,y = y, orientation = 'h')
        return fig 
         
    def ROC_AUC(self, X):
        X['AUC'] = 'AUC'
        
        score = round(auc(X['FPR'],X['TPR']),2)
        fig = px.area(X, x="FPR", y="TPR", color="AUC")
        fig.add_scatter(x=X['threshold'], y=X['threshold'], mode='lines', name = 'threshold' )
        fig.add_annotation(dict(font=dict(color='black',size=10),
                                        x=1.05,
                                        y=0.15,
                                        showarrow=False,
                                        text="AUC = {}".format(score),
                                        textangle=0,
                                        xanchor='left',
                                        xref="paper",
                                        yref="paper"))
        return fig 
        # fig.show()
        # fig.write_image("./AUC.jpeg", engine="kaleido")
    
if __name__=='__main__':    
    
    print(os.getcwd())
    start = time.time()
    if os.path.exists(r'變數篩選.csv'):
        data = pd.read_csv(r'./變數篩選.csv')
        try:
            data.drop('Unnamed: 0', axis = 1, inplace = True)
        except:
            pass 
    else:
        GetData = retrieve_data()
        業者稽查紀錄, 合併PMDS與非登_KAI = GetData.get_data()
        # 合併PMDS與非登_KAI, 無稽查結果業者, 有稽查結果業者 = 未稽查業者(業者稽查紀錄, 合併PMDS與非登_KAI)
        ddata = 合併PMDS與非登_KAI.copy()
        # data.to_csv(r"./變數篩選.csv")
        # 業者稽查紀錄.to_csv(r"./業者稽查紀錄.csv")
        record_insp = pd.read_csv(r"./業者稽查紀錄.csv")
    
    data['稽查結果_改'] = data['稽查結果'].apply(lambda x: 1 if x == '限期改善' else 0)
    DataPre = DataPreprocessing(測試資料起始年份 = 2021, 
                                Impute_method = Impute_method)
    
    key_cols,con_cols,cat_cols,dis_cols, target_cols = DataPre.get_type_columns(data_ = data)
    all_ = key_cols + con_cols + cat_cols + dis_cols + target_cols
    data = data[all_]
    data = DataPre.convert_datatype(data, con_cols+dis_cols)
    data['資本總額元'] = DataPre.log_transoform(data['資本總額元'])
    
    data_cp = data.copy()
    d = DataPre._fill_miss_value(data_cp)
    
    index_cols = ['業者代碼', '市招名稱', '稽查事件編號']
    columns_feature = DataPre.因子 + ['稽查年份']
    X_train, y_train, X_test, y_test,X_train_encoded,X_test_encoded = DataPre.train_test_split(X = d[columns_feature] )
    
    # select model 
    model = BuildModel()
    result = model.ModelSelect(X_train_encoded, 
                               y_train, 
                               model_selected='xgb',
                               criteria = 'precision')
    model.fit(X_train_encoded, y_train)
    thre_table1 = model.performance(X_test_encoded, y_test)
    print("model selection1:\n", model.CLF)
    # select features ('knn' by default)
    f_select = FeatureSelection(DataPre.連續因子 , DataPre.離散因子, DataPre.類別因子)
    f_select.CLF = model.CLF
    X_train_selected, X_test_selected = f_select.SelectMethod(X_train_encoded, y_train, X_test_encoded, y_test)
    y_pre_prob = f_select.prediction
    print("model selection2:\n", model.CLF)
    # Grid search 
    if IsGridSearch:
        tunning = HyperParameterTunning(criteria='average_precision', model=f_select.CLF)
        best_params_, best_score_  = tunning.GridSearch(X_train_selected,y_train,X_test_selected,y_test)
        BestModel = tunning.grid.best_estimator_
        thre_table2 = tunning.thres_table
        GridResult = pd.DataFrame(tunning.grid.cv_results_).T
        GridResult.insert(0, column = 'measurement', value = GridResult.index.tolist())
        GridResult.index = np.arange(1, GridResult.shape[0]+1)
        GridResult.loc[0] =  ['criteria'] + ['average_precision']*3
        GridResult.sort_index(inplace = True )
    else:
        BestModel = f_select.MODEL
    print("model selection3:\n", model.CLF)
    # summarize result 
    df_prob = data[index_cols+['稽查結果_改']].loc[X_test_selected.index.tolist(),:] 
    df_prob['Predict_Prob'] = y_pre_prob
    year_dict = data[['市招名稱', '稽查年份']].groupby('市招名稱').max()
    df_prob['最近稽查年份'] = df_prob['市招名稱'].map(dict(zip(year_dict.index.tolist()
                                                     ,year_dict['稽查年份'].tolist())))
    graph = GraphPlot()
    graph.ROC_AUC(thre_table2)
    
    pd.DataFrame(X_train_selected.columns.tolist(), columns = ['EncodedFeatures']).to_excel(
        './EncodedFeatures.xlsx', index = False
    )
    
    with open("encoder.pkl", "wb") as f: 
        pickle.dump(DataPre.one, f)
        
    with open('model_%s.pkl'%re.search("\w+\(",str(BestModel)).group().replace('(',""), 'wb') as f2:
        pickle.dump(BestModel, f2)
    
    with pd.ExcelWriter(r"./result.xlsx") as writer:
        df_prob.to_excel(writer, sheet_name = '預測機率')
        f_select.result.to_excel(writer, sheet_name = '因子篩選')
        model.result.to_excel(writer, sheet_name = '模型選擇結果')
        thre_table1.to_excel(writer, sheet_name = '因子選擇前threshold(訓練及驗證)', index = False)
        thre_table2.to_excel(writer, sheet_name = '因子選擇後threshold(測試)', index = False)
        GridResult.to_excel(writer,sheet_name = 'GridSearch' )
    
    end = time.time()
    print("Time spent for whole program execution is %d seconds."%(end - start))
    # xgb.DMatrix(data = X_train, label = y_train)
    
    '''
    a = np.array(list(dict(Counter(data['稽查事件編號'])).items()))
    a[a[:,1].astype(int) > 1].shape
    '''
    
    
