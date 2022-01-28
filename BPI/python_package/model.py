import pandas as pd 
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class Model:
    def __init__(self, X_train, y_train, X_test, y_test, voting ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.voting = voting 
    
    def PerformanceEvaluation(self, ConMat):
        TP, FP, FN, TN = ConMat.ravel()
    #     Precision, Recall, F1,Specificity,NPV,TP, FP, FN, TN
    #     TN = ConMat[1,1]
    #     FP = ConMat[0,1]
    #     FN = ConMat[1,0]
    #     TP = ConMat[0,0]

        Precision = TP/(TP + FP)
        Recall = TP/(TP + FN)
        F1 = 2/(1/Precision + 1/Recall)
        TN_FPTN = FP/(FP + TN) # how manay Auctual No in prediction 
        Specificity = TN/(TN+FP) 
        NPV = TN/(FN+TN)
        return Precision, Recall, F1,  Specificity,NPV, round(TP), round(FP), round(FN), round(TN)

    def VotingModelBuild(self, X_train, y_train, X_test, voting):
        estimator = []
        estimator.append(('DecisionTreeC5', DecisionTreeClassifier(criterion = "entropy"))) # C5.0
        estimator.append(('DecisionTree', DecisionTreeClassifier(criterion = "gini")))
        estimator.append(('RandomForest', RandomForestClassifier())) 
        estimator.append(('NaiveBayes', GaussianNB())) 
        estimator.append(('Logistic', LogisticRegression(max_iter = 150)))
        estimator.append(('ElasticNet', LogisticRegression(penalty = "elasticnet",solver = 'saga', l1_ratio = 0.5, max_iter = 150)))
        estimator.append(('GBC', GradientBoostingClassifier()))
        vote = VotingClassifier(estimators = estimator, voting = voting)
        vote.fit(X_train, y_train)
        y_pred = vote.predict(X_test)
        y_pred_proba = vote.predict_proba(X_train)
        return y_pred, y_pred_proba

    def ModelPerformance(self):
        result = pd.DataFrame(index = ['Accuracy','F1Score', 'Precision','Recall',\
            'Specificity' ,'NPV','TP', 'FP', 'FN', 'TN'],columns = ['result'])
    
        y_pred, y_pred_proba = self.VotingModelBuild(self.X_train, self.y_train, self.X_test, self.voting)
        con_mat = confusion_matrix(self.y_test, y_pred, labels = ['N', 'Y'])
        temp = con_mat
        temp2 = temp[0,1]
        temp[0,1] = temp[1,0]
        temp[1,0] = temp2
        con_mat_pd = pd.DataFrame(temp, index = ["N", "Y"], columns = ['N','Y'])
        Precision, Recall, F1,Specificity,NPV,TP, FP, FN, TN = self.PerformanceEvaluation(con_mat) 
        Accuracy = accuracy_score( self.y_test, y_pred)
        result['result'] = [Accuracy, F1, Precision, Recall,Specificity,NPV,TP, FP, FN, TN]      
        return  result, y_pred, con_mat_pd 

