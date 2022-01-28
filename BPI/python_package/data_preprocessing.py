import pandas as pd 
import numpy as np 
from urllib.request import urlopen 
from bs4 import BeautifulSoup
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_graphviz, export_text
import re

class binning:
    
    def __init__(self, x, y, continuous, discrete, categorical_col, Additional,bins = 'custom'):
        self.continuous = continuous
        self.discrete = discrete 
        self.categorical_col = categorical_col 
        self.Additional = Additional
        self.df_x = x 
        self.df_y = y 
        self.bins = bins 
        self.country_map = self.get_country_mapping()
        # self.cut_bins = [self.specified_variable() if bins != 'custom' else  ][0]
        
    def Binning(self, col,x, y ):
        # x = self.d[0]
        # y = self.d[1]
        tdc = DecisionTreeClassifier()
        params = {'max_depth':[2,3,4], 'min_samples_split':[2,3,5,10]}
        Grid = GridSearchCV(tdc, param_grid=params, scoring = 'accuracy')
        Grid.fit(x,y)
        max_depth = Grid.best_params_['max_depth'] 
        min_samples_split = Grid.best_params_['min_samples_split']
        
        # print("Columns =  {}, max_depth = {}, min sample split = {}, \nunique =  {}"\
        #     .format( col, max_depth, min_samples_split, len(list(pd.Series(x[col]).unique()))))
        # print("[error]", type(y),set(y))
        if "Y" in list(set(y)):
            class_weight = {"Y":1.0, "N":5.6}
            tdcs = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, class_weight=class_weight)
            tdcs.fit(x,y)
            tree_rules = export_text(tdcs, feature_names=list(x.columns))
            tree_ = tdcs.tree_
        else:
    #         print("the target do not contain 'Y' and 'N'. You may have to change the binning spltting points. ")
            tdcs = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
            tdcs.fit(x,y)
            tree_rules = export_text(tdcs, feature_names=list(x.columns))
            tree_ = tdcs.tree_
        # print("Threshold:\n", tree_.threshold )
        # print("children_left:\n", tree_.children_left)
        # print("children_right:\n",tree_.children_right)
        # a = tdcs.tree_.threshold
        # print(tree_rules)
        return tree_rules, max_depth, min_samples_split
    
    def CreateLetter(self):
        English_letter = [chr(g) for g in range(65,91)]
        comp_letter = [] 
        for i in list(combinations(English_letter, 2)):
            comp_letter.append(i[0]+i[1]) 
        return comp_letter

    def CreateLabel(self, x,bin_list, letter):
        label = []   
        for j in range(len(bin_list)-1): 
            label.append(letter+str(j)) 
    #     print(label) 
        return label
    
    def specified_variable_bins(self):
        cut_bins = \
            {
            '台幣完稅價格':[0,9843.5,22736.6,28624.5],
            '總淨重公斤數':[0.05, 1588.5, 1677.5],
            "GDP": [1556977745920,2956038111232,7472242229248],
            '全球糧食安全指標':[50.6,59.3,75.65],
            '剩餘效期比例':[0.88,0.89,1],
            '清廉印象指數':[33.5,35.5,36.5,38.75,59.5,69.5,83.5],
            '人類發展指數':[0.55,0.65,0.91],
            '合法權利指數':[3.5,5.5,6.5],
            '全球糧食安全指標_支付能力':[49.75,64.35,79.6],
            '全球糧食安全指標_供應能力':[53,58.5,69.55],
            '全球糧食安全指標_品質及安全':[42.05,57,80.85],
            '進口商收貨未準時申報率':[0.0,0.01,0.94],
            '進口商累積平均總淨重公斤數':[0.5,234.5,4697.5],
            '進口商累積平均台幣完稅價格':[5253.5,26195,26654.5],
            '成立年數':[0.5,37.5],
            '進口商累積報單數':[12.5,26.5,94.5],
            '業別數':[0.5,1.5],
            '下游數量':[0.5,48.5],
            '進口商食品業營業項目數':[0.5,15.5],
            '進口商距前次輸入天數':[9.5,44.5,47.5],
            '進口商中分類累積報單數':[23.5,26.5,484],
            '進口商貨品分類號列累積報單數':[2.5,32.5,74.5],
            '進口商累積報關行家數':[2.5,4.5,6.5],
            '製造廠累積報單數':[4.5,259.5,379.5],
            '製造廠累積總中分類數':[0.5,14.5,24.5],
            '製造廠累積貨品號列數':[0.5,70.5,90],
            '製造廠累積中文標示及外觀不合格數':[1.5,4.5,22.5],
            '製造廠累積報驗註銷數':[0.5,4.5,38.5],
            '製造廠累積扣押放棄退運批數':[3.5,28.5,42.5],
            '報關行累積總中分類數':[41.5,42.5,46.5],
            '逾期交貨數':[0.5,3],
            '稽查GHP複查不合格次數':[1.5,2.5,3.5],
            '稽查HACCP初查不合格次數':[0.5,2.5],
            '稽查HACCP複查次數':[0.5,1.5],
            '稽查HACCP複查不合格次數':[0.5]
            }
        return cut_bins
    
        
    def custom_bins(self, saved_tree):
        splitting_result = {}
        temp_list = []
        temp_list2 = []
        for j in saved_tree:
            temp  = j.split("\n")
            counter = 0
            temp_list = []
            for i in temp:    
                i = re.sub("[\|]|-*|:|<|>|=|\s", "",i)
                if 'class' not in i :
                    if len(i) != 0:
                        column = re.findall('\D+',i)[0]
                        i = re.findall('\d+\.*\d+|\d+',i)
                        if len(i) != 0: 
                            temp_list.append(float(i[0].replace(".", ""))/100)
                            # temp_list.append(i[0])
                        splitting_result[column] = list(set(temp_list))
        return splitting_result
    
    def Binning_transform(self):
        df = self.df_x
        convert_col = self.continuous + self.discrete 
        all_col = self.continuous + self.discrete + self.categorical_col + self.Additional
        # print("before transformation: " , df['受理日期'])
        bin_result = {}
        bin_list_dict = {}
        temporary_df = pd.DataFrame(index = all_col,columns =['max_depth', 'min_samples_split'] )
        comp_letter = self.CreateLetter()
        counter = 0 
        saved_tree = []
        for i in convert_col:
            x1 = pd.DataFrame(df[i])
            y1 = self.df_y
            d = [x1, y1]
            t, max_depth, min_samples_split = self.Binning( i, x1, y1 )
            bin_result[i] = t 
            temporary_df['max_depth'].loc[temporary_df.index == i] = max_depth
            temporary_df['min_samples_split'].loc[temporary_df.index == i] = min_samples_split
            saved_tree.append(t)
        
        if self.bins != 'custom':
            cut_bins = self.specified_variable_bins()
            # print("specified_variable_bins!")
        else:
            cut_bins = self.custom_bins(saved_tree)
            # print("custom_bins!")
        # print('cut_bins: ', cut_bins)
        for j in cut_bins:
            # print(j)
            # bin_list = [min(df[j])-1] + cut_bins[j] + [max(df[j])+1]
            bin_list = []
            bin_list = cut_bins[j]
            bin_list = [min(list(set(self.df_x[j])))] + bin_list + [max(list(set(self.df_x[j])))]
            bin_list = sorted([float(i) for i in list(set(bin_list))])
            label = self.CreateLabel(self.df_x[j], bin_list,comp_letter[counter])
            cut_bins_length = len(bin_list)-1 # when bin label length has to be one fewer than bin edges 
            # you add first element and last elemnet to "bin_list", but there might be duplcate value in the list
            # print("bin labels = {}, bin edges = {}".format(len(English_letter[:cut_bins_length]),len(bin_list)))
            col_identified = []
            df[j] = pd.cut(x = self.df_x[j], bins=bin_list, include_lowest = True, right = True, duplicates = 'drop', \
                labels = label)
            bin_list_dict[j] = bin_list
            counter += 1 
        # print("After transformation: ", type(df['受理日期'].iloc[0]),df['受理日期'])
        
        return df , saved_tree, temporary_df
    
    def get_country_mapping(self):
        # print("Web sraper start!")
        url = urlopen("https://baike.baidu.hk/item/%E4%B8%96%E7%95%8C%E5%90%84%E5%9C%8B/10915534")
        page_html = url.read()
        url.close()
        soup = BeautifulSoup(page_html, 'html.parser')
        country_map = {}
        wano = {}
        benchmark = soup.find_all("div",attrs={"class": "anchor-list"} )
        counter = 0
        for i in soup.find_all("table"):
            ben = benchmark[counter]
            counter += 1 
            for j in i.find_all("tr"):
                prefix = re.findall("[\u4E00-\u9FFF]+",str(ben))[0]
                a = j.text
                for g in j.find_all("td"):
                    w = g.text
                    e = re.sub(g.text,"",a)
                    if len(g.text) != len(a):
                        e = prefix + "_" + e
                    else:
                        e = prefix
                    k = g.text.split("、")
                    for element in k:
                        element = re.sub('\s',"",element)
                        country_map[element] = prefix
                        wano[element] = e
        # print("Web sraper end!")
        return country_map 
    
    def missing_country(self, x):
        mssing_map1 = {"瓜地馬拉":"危地馬拉", '台灣,中華民國':"中華人民共和國","義大利":"意大利", '克羅埃西亞':'克羅地亞',
                    '印尼':"印度尼西亞", '柬埔寨王國':"柬埔寨", "波士尼亞":"歐洲", '馬其頓':"歐洲", '阿拉伯聯合大公國':"沙特阿拉伯",
                    '敘利亞':"敍利亞","紐西蘭":"新西蘭", "象牙海岸":"科特迪瓦",'奈及利亞':"尼日利亞", '大溪地':"亞洲",\
                    "肯亞":"肯尼亞","巴布亞紐幾內亞":"巴布亞新幾內亞","克羅地亞":"克羅地亞"}
        mssing_map2 = {'香港':"中華人民共和國"}
        mssing_map3 = {'中國大陸':"中華人民共和國"}
        if x in list(mssing_map1.keys()):
            return mssing_map1[x]
        elif x in list(mssing_map2.keys()):
            return mssing_map2[x]
        elif x in list(mssing_map3.keys()):
            return mssing_map3[x]
        else:
            return x
        
    def missing_country2(self, x):
        
        if x in list(self.country_map.keys()):
            return self.country_map[x]
        else:
            return x
        
    def country_binning(self, data):
        df = data 
        # print("country_binning begins.")
        d = pd.read_excel(r"C:\Users\bbkelly\Desktop\Kelly\BPI 效益資料\20211210模型重跑結果\生產國別資料中代碼.xlsx")
        first_map = {}
        for c in range(d.shape[0]):
            first_map[d['國別代碼'].iloc[c]] = d['生產國別'].iloc[c]
        data['生產國別'] = data['生產國別'].map(first_map)
        # print("s1:", data['生產國別'].unique())
        data['生產國別'] = data['生產國別'].apply(self.missing_country)
        # print("s2:", data['生產國別'].unique())
        data['生產國別'] = data['生產國別'].apply(self.missing_country2)
        # print("s3:", data['生產國別'].unique())
        if True in list(data['生產國別'].isnull()):
            print("There are null values!")
            print(data[data['生產國別'].isnull()]['生產國別'].unique())
        else:
            print("No null values!")
            print(data['生產國別'].unique())
        # print(data.info())
        return data
    