import numpy as pd 
import pandas as pd 
from scipy.stats import chi2_contingency
from sklearn import feature_selection
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2 
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
import numpy as np 
import re 
from python_package.data_check import CheckData


class CodeSplitting:
    
    def __init__(self, data, x, y, continuous,discrete,categorical_col,split_method = 'kmeans', data_continuous = None):
        self.check = CheckData()
        self.df = data 
        self.df['受理日期'] = self.df['受理日期'].apply(self.check.String_to_datetime_sorted_apply)
        self.x = x 
        self.y = y 
        self.data_continuous = data_continuous
        self.continuous = continuous 
        self.discrete = discrete
        self.categorical_col = categorical_col
        self.split_method = split_method
        self.selected_col, self.best_param = self.feature_selection()
    
    def remove_ending(self, x):
        if re.findall("\.0", x):
            x = re.sub("\.0","" ,x)
            if len(x) == 10:
                x = '0' + x 
            elif len(x) == 11:
                x = x 
            return x    
        
    def correlation_check(self, x, y ):
        corr_chi = {}
        for i in x:
            try:
                CrossTable = pd.crosstab(self.x[i], self.y)
                chi2, p, dof, ex = chi2_contingency(CrossTable)
                if p < 0.05:
                    corr_chi[i] = round(p, 5)
            except:
                print('[Error]', i)
        return corr_chi

    def pca_corr(self):
        pca = PCA()
        x_continue = self.data_continuous
        X_ = StandardScaler().fit_transform(x_continue)
        X_ = pca.fit_transform(x_continue)
        X_pca = pd.DataFrame(data = X_ , columns = [ 'PC' + str(i) for i in range(X_.shape[1])])
        pca_result = pd.DataFrame(pca.components_, columns = self.data_continuous.columns,\
            index = [ 'PC' + str(i) for i in range(X_.shape[1])])
        # pd.DataFrame(pca.explained_variance_ratio_).plot.bar()
        # plt.legend('')
        # plt.xlabel('Principal Components')
        # plt.ylabel('Explained Varience')
        df_variance_explained = pd.DataFrame(pca.explained_variance_ratio_)
        return X_pca,pca_result,pd.DataFrame(pca.explained_variance_ratio_)

    def feature_selection(self):
        label = LabelEncoder()
        ordinal = OrdinalEncoder()
        y_ = np.array(label.fit_transform(self.y))
        x_ = np.array(ordinal.fit_transform(self.x))
        sf = SelectKBest(chi2, k='all')
        sf_fit1 = sf.fit(x_, y_)
        params = {'k':[5,10,15,20,25,'all']}
        Grid = GridSearchCV(sf, param_grid=params, scoring = 'accuracy')
        Grid.fit(x_, y_)
        best_param = Grid.best_params_['k']
        sf_fit1.scores_
        score_df = pd.DataFrame(columns = ['Score'], index = self.x.columns)
        score_df = score_df.sort_values(by = ['Score'], ascending = False)
        selected_col = list(score_df.index[:best_param]) + ['檢驗結果_重製']
        return selected_col, best_param
     
    def code_index(self, code):
        df = self.df.copy()
        code['code'] = code['code'].astype(str)
        r = np.zeros(len(list(set(code['code'])))*4).reshape(-1,4)
        df_code = pd.DataFrame(r, index = list(set(code['code'].astype(str))), columns = list(np.arange(3)) + ['result'])
        for i in list(set(code['code'])):
            count = code[code['code']==i].shape[0]
            group_list = list(code[code['code']==i]['group'])
            for j in group_list:
                df_code.loc[i,j] = df_code.loc[i,j] + 1 
        for g in range(df_code.shape[0]):
            df_code['result'].iloc[g] = np.argmax(df_code.iloc[g,:])
        group = {}
        n = len(list(code['group'].unique()))
        for i in range(n):
            group[i] = list(df_code[df_code['result']==i].index)
        
        df_group = dict(zip(np.arange(n), [""]*n))
        
        for j in range(3):
            for i in group[j]:
                tmpe_df = pd.DataFrame()
                temp = df[df['貨品分類號列']==i]
                if isinstance(df_group[j], str) :
                    df_group[j] = temp 
                else: 
                    df_group[j] = pd.concat([df_group[j], temp],axis = 0)

        x1 = df_group[0]
        x2 = df_group[1]
        x3 = df_group[2]
        
        return [x1,x2,x3]
     
    def determine_n_clusters(self):
        try:
            x_code_bin = self.Binning_transform(self.x, self.y, self.continuous, self.discrete,self.categorical_col)
        except:
            x_code_bin = self.x
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(x_code_bin[self.selected_col])
        x_code_bin_encode = pd.DataFrame(encoder.transform(x_code_bin[self.selected_col]).toarray(),\
                                        index = x_code_bin[self.selected_col].index)
        x_code_bin_encode.columns = encoder.get_feature_names()
        
        if self.split_method == 'kmeans':
            code = pd.DataFrame(columns = ['code', 'group'], index = np.arange(self.y.shape[0]))
            Sum_of_squared_distances = []
            K = range(1,15)
            for k in K:
                km = KMeans(n_clusters=k)
                km = km.fit(x_code_bin_encode)
                Sum_of_squared_distances.append(km.inertia_)
            # plt.figure(figsize = (15,10))
            # plt.plot(K, Sum_of_squared_distances, 'bx-')
            # plt.xlabel('k')
            # plt.ylabel('Sum_of_squared_distances')
            # plt.title('Elbow Method For Optimal k')
            # plt.show()
            # num = int(input("Please enter number of clusters based on the line graph:"))
            num = 3 
            KM = KMeans(n_clusters=num)
            KM.fit_transform(x_code_bin_encode)
            for i in range(len(KM.labels_)):
                code.iloc[i,:] = [self.y.iloc[i], KM.labels_[i]]
            code['code'] = code['code'].astype(str)
            all_groups = self.code_index(code)
        
        elif self.split_method == "existing":
            df_product = pd.read_excel(r"香辛料中分類拆分.xlsx")
            df_product['貨品分類號列中文名稱(香辛料)'] = df_product['貨品分類號列中文名稱(香辛料)'].fillna(method = "ffill")
            df_product['貨品分類號列'] = df_product['貨品分類號列'].apply(self.check.remove_ending_code)
            length = len(list(df_product['貨品分類號列中文名稱(香辛料)'].unique()))
            all_groups = []
            code = df_product[['貨品分類號列','貨品分類號列中文名稱(香辛料)' ]]
            code.columns = ['code1', 'group1']
            counter = 0
            for i in df_product['貨品分類號列中文名稱(香辛料)'].unique():
                a = df_product[df_product['貨品分類號列中文名稱(香辛料)']==i]['貨品分類號列']
                temp_df = pd.DataFrame() 
                for j in a:   
                    temp = self.df[self.df['貨品分類號列']==str(j)]
                    temp_df = pd.concat([temp_df, temp],axis = 0)
                all_groups.append(temp_df)
        # print("determine_n_clusters-all_groups = ", len(all_groups))
        return all_groups, code
        
    def split_data_into_groups(self):
        if self.split_method == 'kmeans':
            print("Error0!")
            all_group_list,code = self.determine_n_clusters()     
        elif self.split_method == "existing":
            all_group_list,code = self.determine_n_clusters()
        
        if len(all_group_list)==3:
            print("Error1!")
            x1, x2, x3 = all_group_list[0],all_group_list[1],all_group_list[2]
            df1_x, df1_y = x1.drop(['檢驗結果_重製'],axis = 1), x1['檢驗結果_重製']
            df2_x, df2_y = x2.drop(['檢驗結果_重製'],axis = 1), x2['檢驗結果_重製']
            df3_x, df3_y = x3.drop(['檢驗結果_重製'],axis = 1), x3['檢驗結果_重製']
            return df1_x,df1_y, df2_x, df2_y, df3_x, df3_y,code
        
        elif len(all_group_list) == 4:
            x1, x2, x3,x4 = all_group_list[0],all_group_list[1],all_group_list[2],all_group_list[3]
            df1_x, df1_y = x1.drop(['檢驗結果_重製'],axis = 1), x1['檢驗結果_重製']
            df2_x, df2_y = x2.drop(['檢驗結果_重製'],axis = 1), x2['檢驗結果_重製']
            df3_x, df3_y = x3.drop(['檢驗結果_重製'],axis = 1), x3['檢驗結果_重製']
            df4_x, df4_y = x4.drop(['檢驗結果_重製'],axis = 1), x4['檢驗結果_重製']
            return df1_x,df1_y, df2_x, df2_y, df3_x, df3_y,df4_x, df4_y,code
    

