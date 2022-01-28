import pandas as pd 
import numpy as np 
from imblearn.over_sampling import SMOTENC, SMOTE, SMOTEN
from python_package.data_check import CheckData
from sklearn.preprocessing import OneHotEncoder

class smote_transform:
    
    def __init__(self, categorical_col, class_weight,data,**TrainTestGroup):
        self.categorical_col = categorical_col
        self.class_weight = class_weight
        self.data = data
        self.TrainTestGroup = TrainTestGroup
         
        
    def custom_smote(self, X_train, y_train,  algo,sampling_strategy, *cat):
        cat = pd.Series(cat)
        a = dict(zip(list(X_train.columns), np.arange(X_train.shape[0])))
        cat = list(cat.map(a))
        if algo == "smotenc":
            over_sample = SMOTENC(categorical_features = cat,sampling_strategy = sampling_strategy)    
        elif algo == "smote":
            over_sample = SMOTE(sampling_strategy = sampling_strategy)  
        elif algo == 'smoten':
            over_sample = SMOTEN(sampling_strategy = sampling_strategy)  
        else:
            return "please specify algo argument"
        
        X_train.index = np.arange(X_train.shape[0])
        X_smote, y_smote = over_sample.fit_resample(X_train,y_train)
        
        return X_smote, y_smote
    
    def smote_encode(self):
        Smote_TrainTestSplit = {}
        self.TrainTestGroup = self.TrainTestGroup['TrainTestGroup']
        counter = 0 
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(self.data)
        # print("smote come in. ")
        # print(self.TrainTestGroup)
        for group, data_ in self.TrainTestGroup.items():
            # print("group:" , group)
            # print('data_ key: ' , data_.keys())
            # print(data_)
            counter += 1 
            xtrain = data_['X_train']
            ytrain = data_['y_train']
            xtest = data_['X_test']
            ytest = data_['y_test']
            print("[before smote] {}: xtrain: {}, ytrain:{}, xtest:{}, ytest:{}"\
                .format(group, xtrain.shape,ytrain.shape, xtest.shape, ytest.shape))
            # print("生產國別unique: ",self.TrainTestGroup['data']['X_train']['生產國別'].unique())
            X_smote, y_smote = self.custom_smote(xtrain,ytrain, 'smoten',self.class_weight,*self.categorical_col) 
            print("{} {}th\n{}\n{}".format(group,  str(counter), X_smote.index,X_smote ))
            print("")
            print("X_smote:\n", X_smote.head())
            # print(X_smote['生產國別'].unique())
            print("")
            X_train_encode = pd.DataFrame(encoder.transform(X_smote).toarray(), index = X_smote.index)
            X_train_encode.columns = encoder.get_feature_names()
            X_test_encode = pd.DataFrame(encoder.transform(xtest).toarray(), index = xtest.index)
            X_test_encode.columns = encoder.get_feature_names()
            Smote_TrainTestSplit[group] = {"X_train": X_smote, "y_train":y_smote,\
                                            "X_test":xtest , "y_test":ytest }
            # X_train_encode = pd.DataFrame(encoder.transform(xtrain).toarray(), index = xtrain.index)
            # X_train_encode.columns = encoder.get_feature_names()
            # X_test_encode = pd.DataFrame(encoder.transform(xtest).toarray(), index = xtest.index)
            # X_test_encode.columns = encoder.get_feature_names()
            Smote_TrainTestSplit[group] = {"X_train": X_train_encode, "y_train":y_smote,\
                                               "X_test":X_test_encode , "y_test":ytest }
        return Smote_TrainTestSplit

