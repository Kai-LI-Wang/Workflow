
from sqlalchemy import all_
from python_package.DB import *
import warnings
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from python_package.data_preprocessing import *
from  python_package.Kmeans_splitting import CodeSplitting
from python_package.data_check import CheckData 
from python_package.TrainTestSplit import TTSplitting
from python_package.SMOTE import smote_transform
from python_package.model import Model 
import datetime as dt 
from collections import Counter
warnings.filterwarnings("ignore")



# if __name__=='__main__':

checkdata = CheckData()
continuous,Additional,categorical_col,discrete = checkdata.continuous, checkdata.Additional, \
                                                 checkdata.categorical_col, checkdata.discrete

fd = open(r'C:\Users\bbkelly\Desktop\Kelly\BPI 效益資料\20211210模型重跑結果\sql\原始資料.sql', 'r')
sqlFile = fd.read()
fd.close()

start = dt.datetime.now()
a = retrieve_data_sql(sqlFile)
data_sql = a.query_func()
raw_data = cleaning_data(data_sql).data2
spare_data = raw_data.copy()

spare_data['貨品分類號列'] = spare_data['貨品分類號列'].apply(checkdata.remove_ending_code )

#---
raw_code_y = spare_data['貨品分類號列']
raw_code_x = spare_data[continuous + Additional + categorical_col + discrete]

binn_class = binning( raw_code_x, raw_code_y, continuous, discrete, categorical_col, Additional)
binn_X, division_rule, tree_param = binn_class.Binning_transform()
k = binn_X.copy()
g = binn_X.copy()
h = binn_class.country_binning(g)
# country_name_map = {'亞洲':'AS', '北美洲':'NA', \
#     '非洲':'AF', '歐洲':'EU','南美洲':'SA', '大洋洲':'OC' }
# k = h.copy()
# k['生產國別'] = k['生產國別'].map(country_name_map)

end = dt.datetime.now()
print("total time spent: ", end - start)

temp_df = k.copy()
temp_df['檢驗結果_重製'] = raw_data['檢驗結果_重製']


for Methods in ["kmeans", 'existing']:
    kmeans_split_class = CodeSplitting(\
                                        temp_df,\
                                        x = k,\
                                        y = raw_code_y,\
                                        continuous = continuous,\
                                        discrete = discrete,\
                                        categorical_col = categorical_col,\
                                        split_method = Methods)
    if Methods == 'kmeans':
        selected_col, score_df = kmeans_split_class.feature_selection()
        Kdf1_x,Kdf1_y, Kdf2_x, Kdf2_y, Kdf3_x, Kdf3_y,Kcode = kmeans_split_class.split_data_into_groups()
        grouping_kmeans = [Kdf1_x,Kdf1_y, Kdf2_x, Kdf2_y, Kdf3_x, Kdf3_y]
    elif Methods == 'existing': 
        Adf1_x,Adf1_y, Adf2_x, Adf2_y, Adf3_x, Adf3_y,Adf4_x, Adf4_y,Acode = kmeans_split_class.split_data_into_groups()
        grouping_existing = [Adf1_x,Adf1_y, Adf2_x, Adf2_y, Adf3_x, Adf3_y,Adf4_x, Adf4_y]
      
col  =  continuous+discrete+categorical_col+['貨品分類號列', '受理日期']     
x = k[col]
# num = ['是否有商標', '是否為代理進口', '有無報驗代理人', '黑名單產品', '黑名單進口商', '無製造日期', '無有效日期']
# for i in col:
#     if i in num:
#         x[i] = x[i].astype(str)
#     else:
#         x[i] = x[i].astype(str)



x = checkdata.String_to_datetime_sorted(x)
y = raw_data['檢驗結果_重製']
counter = 0 
c = 0 
kgroup = {}
egroup = {}
for group in [grouping_kmeans,grouping_existing]:
    counter = 0
    for i in range(0,len(group), 2):
        counter += 1
        if c == 0:
            kgroup['data'+str(counter)] = [group[i] ,group[i+1]]
            if counter == 3 :
                kgroup['data'] = [x,y] 
        elif c == 1:
            egroup['data'+str(counter)] = [group[i] ,group[i+1]]
            if counter == 4:
                egroup['data'] = [x,y] 
    c += 1 

split = TTSplitting()
edata = egroup['data'][0][egroup['data'][0].columns[:-2]]
kdata = kgroup['data'][0][kgroup['data'][0].columns[:-2]]
TrainTestSplitE = split.TrainTestSplitFunction( custom=True,TestSize=None, **egroup)
TrainTestSplitK = split.TrainTestSplitFunction( custom=True,TestSize=None,**kgroup)
smoteE = smote_transform(categorical_col = categorical_col, \
                    class_weight = 3/7, data = edata,TrainTestGroup = TrainTestSplitE)
SmoTrainTestSplitE = smoteE.smote_encode()
smoteK = smote_transform(categorical_col = categorical_col, \
                    class_weight = 3/7, data = kdata,TrainTestGroup = TrainTestSplitK)
SmoTrainTestSplitK = smoteK.smote_encode()


# for i in edata:
#     edata[i] = edata[i].astype(str)

# a = SmoTrainTestSplitE.copy()
# result  = {}
# encoder = OneHotEncoder(handle_unknown='ignore')
# encoder.fit(edata)

# for group, data_ in a.items():
#     x_train = a[group]['X_train']
#     x_test = a[group]['X_test']
#     y_train = a[group]['y_train']
#     y_test = a[group]['y_test']

#     # print(data_)
#     a[group]['X_train'] = pd.DataFrame(encoder.transform(x_train).toarray())
#     a[group]['X_train'].columns = encoder.get_feature_names()
    
#     a[group]['X_test'] = pd.DataFrame(encoder.transform(x_test).toarray())
#     a[group]['X_test'].columns = encoder.get_feature_names()
    
  
start = dt.datetime.today()
ModelResult = {}
# group_all = [group_smote_encode_model, all_smote_encode_model]
all_result = {}
all_pred_value = {}
t = ['group_', 'all_']

con_mat_dict = {}
n = 0
for DataGroup in [SmoTrainTestSplitE,SmoTrainTestSplitK]:
    n += 1 
    counter = 0 
    for key, value in DataGroup.items():
        print(key, value['X_train'].shape,value['y_train'].shape,value['X_test'].shape,value['y_test'].shape)
        for value2 in list(DataGroup.values())[1:]:
            if key == 'data':
                counter += 1 
                print(value['X_train'].shape,value['y_train'].shape, value2['X_test'].shape,value2['y_test'].shape)
                model = Model(value['X_train'],value['y_train'],value2['X_test'],value2['y_test'],'soft')
                result, y_pred,con_mat_pd = model.ModelPerformance()
                ModelResult['AllData'+ str(counter)] = result 
                all_pred_value['AllData'+ str(counter)] = y_pred
                con_mat_dict['AllData'+ str(counter)] = con_mat_pd 
            else:
                print(value['X_train'].shape,value['y_train'].shape, value2['X_test'].shape,value2['y_test'].shape)
                model = Model(value['X_train'],value['y_train'],value2['X_test'],value2['y_test'], 'soft')
                result, y_pred,con_mat_pd = model.ModelPerformance()
                ModelResult[key] = result 
                all_pred_value[key] = y_pred 
                con_mat_dict[key] = con_mat_pd 
    all_result[n] = ModelResult

end = dt.datetime.today()
print("Time spent: ", end - start)




