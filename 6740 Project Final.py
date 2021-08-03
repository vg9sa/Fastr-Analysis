#!/usr/bin/env python
# coding: utf-8

# In[249]:


#NFL data Analysis
#ISYE 6740 Project

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import scipy.io
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import datasets, svm, metrics
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay

# In[130]:


pd.options.display.max_columns = 100
pd.options.display.max_rows = None


# In[226]:


def plot_decision_boundary(model, title, x_train, x_test, y_train):
    h = .5
    cmap_light = ListedColormap(['#FFAAAA',  '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000',  '#0000FF'])

    x_min, x_max = x_train[:,0].min(), x_train[:,0].max() 
    y_min, y_max = x_train[:,1].min(), x_train[:,1].max()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

    # Also plot the training points
    scatter = plt.scatter(x_train[:,0], x_train[:,1], c=y_train, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.legend(handles=scatter.legend_elements()[0] , labels= ('0', '1'))


# In[156]:


#loading data
years = [2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006]

#Create Lists for Game level and Season Level Data
combined_data_GM = []
testingData_GM = []
combined_data_SE = []
testingData_SE = []

#Processing Loop - Starts at Game Level and then aggregates to Season Level
for year in years:
    data = pd.read_csv('https://github.com/guga31bb/nflfastR-data/blob/master/data/'                                  'play_by_play_' + str(year) + '.csv.gz?raw=True',
                                 compression='gzip', low_memory=False)
    data.drop(['passer_player_name', 'passer_player_id',
                   'rusher_player_name', 'rusher_player_id',
                   'receiver_player_name', 'receiver_player_id'],
                  axis=1, inplace=True)
    data = data.loc[(data.play_type.isin(['no_play','pass','run','extra_point','field_goal','punt'])) & (data.epa.isna()==False)]
    data.play_type.loc[data['pass']==1] = 'pass'
    data.play_type.loc[data.rush==1] = 'run'
    data['is_home_team'] = np.where(data['home_team']== data['posteam'], 1, 0)
    data['fg_made'] = np.where(data['field_goal_result']== 'made', 1, 0)
    data['fg_missed'] = np.where(data['field_goal_result']== 'missed', 1, 0)
    data['year'] = str(year)
    data.fillna(0)
    data.reset_index(drop=True, inplace=True)
    nflData = data.copy()
    playoffData = nflData[nflData['season_type']=='POST']
    regSznData = nflData[nflData['season_type']=='REG']
    playoff_teams = np.unique(playoffData[['home_team','away_team']].values).tolist()
    allTeams = np.unique(regSznData[['home_team','away_team']].values).tolist()
    
    features = ['year','week','game_id','posteam','yardline_100','yards_gained','score_differential','epa','passing_yards','rushing_yards','penalty_yards','cpoe','pass','rush','fg_made','fg_missed','is_home_team','first_down','third_down_failed','touchdown','interception','sack','tackled_for_loss','fumble']
    featureData = regSznData[features]
    featureData2 = featureData.groupby(['year','week','game_id','posteam'], as_index = False).agg({'yardline_100': 'mean',
                                                                                                'yards_gained': 'sum',
                                                                                                'score_differential': 'mean',
                                                                                                'epa': 'mean',
                                                                                                'passing_yards': 'sum',
                                                                                                'rushing_yards': 'sum',
                                                                                                'penalty_yards': 'sum',
                                                                                                'cpoe': 'mean',
                                                                                                'pass': 'sum',
                                                                                                'rush': 'sum',
                                                                                                'fg_made': 'sum',
                                                                                                'fg_missed': 'sum',
                                                                                                'is_home_team': 'mean',
                                                                                                'first_down': 'sum',
                                                                                                'third_down_failed': 'sum',
                                                                                                'touchdown': 'sum',
                                                                                                'interception': 'sum',
                                                                                                'sack': 'sum',
                                                                                                'tackled_for_loss': 'sum',
                                                                                                'fumble': 'sum'})
    #Playoff Check
    def playoffCheck(x):
        if x in playoff_teams:
            return 1
        else:
            return 0

    featureData2['playoffs'] = featureData2.apply(lambda row: playoffCheck(row['posteam']),axis=1)
    
    #Dump Game Level Data
    if year == 2020:
        testingData_GM.append(featureData2)
    elif year == 2019:    
        testingData_GM.append(featureData2)
    elif year == 2018:
        testingData_GM.append(featureData2)
    else:
        combined_data_GM.append(featureData2)
    
    #Reprocess for Season Level
    featureData2.drop(['year','week','game_id'], axis = 1)
    featureData3 = featureData2.groupby(['posteam'], as_index = False).mean()
    
    #Dump Season Level Data
    if year == 2020:
        testingData_SE.append(featureData3)
    elif year == 2019:    
        testingData_SE.append(featureData3)
    elif year == 2018:
        testingData_SE.append(featureData3)
    else:
        combined_data_SE.append(featureData3)

#Game level Aggregation
finalData_GM = pd.concat(combined_data_GM).reset_index(drop=True)
testingFinalData_GM = pd.concat(testingData_GM).reset_index(drop=True)

#Season Level Aggregation
finalData_SE = pd.concat(combined_data_SE).reset_index(drop=True)
testingFinalData_SE = pd.concat(testingData_SE).reset_index(drop=True)


# In[157]:


finalData_GM


# In[158]:


finalData_SE


# In[159]:


#Data Split
X_train,y_train = finalData_SE.iloc[:,2:22],finalData_SE.iloc[:,22]
X_test, y_test = testingFinalData_SE.iloc[:,2:22], testingFinalData_SE.iloc[:,22]


# In[175]:


#Logistic Regression
logit = LogisticRegression(max_iter=10_000)
clf =logit.fit(X_train, y_train)
prediction = clf.predict(X_test)
print("Logistic Regression:")
print("Classification report for classifier %s:\n%s\n"
      % (logit, metrics.classification_report(y_test, prediction)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))
resultLR = confusion_matrix(y_true=y_test, y_pred=prediction, labels=[0,1])
accuracy_LR = np.sum(np.diagonal(resultLR))/np.sum(resultLR)
print ("The accuracy of Logistic Regression is "+str(round(accuracy_LR,3)*100)+"%")
#thresholding
thresh_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
master_tester = pd.DataFrame()
preds = clf.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["logit"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)


# In[176]:


# navie bayes with GaussianNB
bayes = GaussianNB()
NB1 = bayes.fit(X_train, y_train)
predictionNB = NB1.predict(X_test)
resultNB = confusion_matrix(y_true=y_test, y_pred=predictionNB, labels=[0,1])
accuracy_NB = np.sum(np.diagonal(resultNB))/np.sum(resultNB)
print("Naive Bayes:")
print("Classification report for classifier %s:\n%s\n"
      % (logit, metrics.classification_report(y_test, predictionNB)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionNB))
print ("The accuracy of Naive Bayes is "+str(round(accuracy_NB,3)*100)+"%")
preds = bayes.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["naive bayes"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)
# In[182]:


#SVM
svm = SVC(kernel = 'linear', probability = True)
SVMR1 = svm.fit(X_train, y_train)
predsvm1 = SVMR1.predict(X_test)
resultsvm1 = confusion_matrix(y_true=y_test, y_pred=predsvm1, labels=[0,1])
accuracy_svm1 = np.sum(np.diagonal(resultsvm1))/np.sum(resultsvm1)
print("SVM - Linear:")
print("Classification report for classifier %s:\n%s\n"
      % (svm, metrics.classification_report(y_test, predsvm1)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predsvm1))
print ("The accuracy of SVM-Linear is "+str(round(accuracy_svm1,3)*100)+"%")
preds = svm.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["SVM"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)
    
# In[186]:


#Kernel SVM
svm2 = SVC(kernel = 'rbf', gamma= 0.01, probability = True)
SVMR2 = svm2.fit(X_train, y_train)
predsvm2 = SVMR2.predict(X_test)
resultsvm2 = confusion_matrix(y_true=y_test, y_pred=predsvm2, labels=[0,1])
accuracy_svm2 = np.sum(np.diagonal(resultsvm2))/np.sum(resultsvm2)
print("Kernel SVM:")
print("Classification report for classifier %s:\n%s\n"
      % (svm2, metrics.classification_report(y_test, predsvm2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predsvm2))
print ("The accuracy of Kernel SVM is "+str(round(accuracy_svm2,3)*100)+"%")
preds = svm2.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Kernel SVM"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[291]:


#Neural Network
nnw = MLPClassifier(hidden_layer_sizes = (20, 10), max_iter=10_000)
nnwR = nnw.fit(X_train, y_train)
prednnw = nnwR.predict(X_test)
resultsnnw = confusion_matrix(y_true=y_test, y_pred=prednnw, labels=[0,1])
accuracy_nnw = np.sum(np.diagonal(resultsnnw))/np.sum(resultsnnw)
print("Neural Network:")
print("Classification report for classifier %s:\n%s\n"
      % (nnw, metrics.classification_report(y_test, prednnw)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prednnw))
print ("The accuracy of Neural Network is "+str(round(accuracy_svm2,3)*100)+"%")
preds = nnw.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Neural Net"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[294]:


#KNN
KNN = knn(n_neighbors=1)
KNN1 = KNN.fit(X_train, y_train)
predictionK = KNN1.predict(X_test)
resultK = confusion_matrix(y_true=y_test, y_pred=predictionK, labels=[0,1])
accuracy_KNN = np.sum(np.diagonal(resultK))/np.sum(resultK)
print("KNN:")
print("Classification report for classifier %s:\n%s\n"
      % (KNN, metrics.classification_report(y_test, predictionK)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionK))
print ("The accuracy of KNN is "+str(round(accuracy_KNN,3)*100)+"%")
preds = KNN.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["KNN"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[295]:


#CART
CART = tree.DecisionTreeClassifier(random_state = 1)
CART1 = CART.fit(X_train, y_train)
predictionCART = CART1.predict(X_test)
resultCART = confusion_matrix(y_true=y_test, y_pred=predictionCART, labels=[0,1])
accuracy_CART = np.sum(np.diagonal(resultCART))/np.sum(resultCART)
print("CART:")
print("Classification report for classifier %s:\n%s\n"
      % (CART, metrics.classification_report(y_test, predictionCART)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionCART))
print ("The accuracy of CART is "+str(round(accuracy_CART,3)*100)+"%")
preds = CART.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["CART"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[299]:


#Random Forest
RF = ensemble.RandomForestClassifier(random_state = 1)
RF1 = RF.fit(X_train, y_train)
predictionRF = RF1.predict(X_test)
resultRF = confusion_matrix(y_true=y_test, y_pred=predictionRF, labels=[0,1])
accuracy_RF = np.sum(np.diagonal(resultRF))/np.sum(resultRF)
print("Random Forest:")
print("Classification report for classifier %s:\n%s\n"
      % (RF, metrics.classification_report(y_test, predictionRF)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionRF))
print ("The accuracy of Random Forest is "+str(round(accuracy_RF,3)*100)+"%")
preds = RF.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Random Forest"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[312]:


#Adaboost
ADB = ensemble.AdaBoostClassifier(random_state = 1)
ADB1 = ADB.fit(X_train, y_train)
predictionADB = ADB1.predict(X_test)
resultADB = confusion_matrix(y_true=y_test, y_pred=predictionADB, labels=[0,1])
accuracy_ADB = np.sum(np.diagonal(resultADB))/np.sum(resultADB)
print("AdaBoost:")
print("Classification report for classifier %s:\n%s\n"
      % (ADB, metrics.classification_report(y_test, predictionADB)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionADB))
print ("The accuracy of AdaBoost is "+str(round(accuracy_ADB,3)*100)+"%")
preds = ADB.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Adaboost"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[320]:


#Gradient Boosting
GB = ensemble.GradientBoostingClassifier(random_state = 1)
GB1 = GB.fit(X_train, y_train)
predictionGB = GB1.predict(X_test)
resultGB = confusion_matrix(y_true=y_test, y_pred=predictionGB, labels=[0,1])
accuracy_GB = np.sum(np.diagonal(resultGB))/np.sum(resultGB)
print("Gradient Boosting:")
print("Classification report for classifier %s:\n%s\n"
      % (GB, metrics.classification_report(y_test, predictionGB)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionGB))
print ("The accuracy of Gradient Boost is "+str(round(accuracy_GB,3)*100)+"%")
preds = GB.predict_proba(X_test)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Gradient Boosting"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)   
    
    
# In[ ]:


#PCA Models


# In[195]:


pca = PCA(n_components=2)


# In[197]:


train_transformed = pca.fit(finalData_SE.iloc[:,2:22]).transform(finalData_SE.iloc[:,2:22])


# In[198]:


testing_transformed = pca.transform(testingFinalData_SE.iloc[:,2:22])


# In[204]:


X_train2,y_train2 = train_transformed,finalData_SE.iloc[:,22]
X_test2, y_test2 = testing_transformed, testingFinalData_SE.iloc[:,22]


# In[ ]:


# navie bayes with GaussianNB PCA


# In[207]:


NB = bayes.fit(X_train2, y_train2)


# In[227]:


plot_decision_boundary(NB, 'Naive Bayes', X_train2, X_test2, y_train2)


# In[219]:


predictionNB2 = NB.predict(X_test2)
resultNB2 = confusion_matrix(y_true=y_test2, y_pred=predictionNB2, labels=[0,1])
accuracy_NB2 = np.sum(np.diagonal(resultNB2))/np.sum(resultNB2)
print('Naive Bayes PCA')
print("Classification report for classifier %s:\n%s\n"
      % (nnw, metrics.classification_report(y_test2, predictionNB2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predictionNB2))
print ("The accuracy of Naive Bayes PCA is "+str(round(accuracy_NB2,3)*100)+"%")
preds = NB.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["NB-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:


# logistic regression PCA


# In[220]:


LR = logit.fit(X_train2, y_train2)


# In[228]:


plot_decision_boundary(LR, 'Logistic Regression', X_train2, X_test2, y_train2)


# In[221]:


predictionLR2 = LR.predict(X_test2)
resultLR2 = confusion_matrix(y_true=y_test2, y_pred=predictionLR2, labels=[0,1])
accuracy_LR2 = np.sum(np.diagonal(resultLR2))/np.sum(resultLR2)
print('Logistic Regression PCA')
print("Classification report for classifier %s:\n%s\n"
      % (nnw, metrics.classification_report(y_test2, predictionLR2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predictionLR2))
print ("The accuracy of Logistic Regression PCA is "+str(round(accuracy_LR2,3)*100)+"%")
preds = LR.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Logit-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[229]:


# KNN PCA


# In[242]:


KNN2 = knn(n_neighbors=1)


# In[243]:


KNNR = KNN2.fit(X_train2, y_train2)


# In[244]:


plot_decision_boundary(KNNR, 'KNN', X_train2, X_test2, y_train2)


# In[245]:


predictionK2 = KNNR.predict(X_test2)
resultK2 = confusion_matrix(y_true=y_test2, y_pred=predictionK2, labels=[0,1])
accuracy_KNN2 = np.sum(np.diagonal(resultK2))/np.sum(resultK2)
print('KNN PCA')
print("Classification report for classifier %s:\n%s\n"
      % (nnw, metrics.classification_report(y_test2, predictionK2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predictionK2))
print ("The accuracy of KNN PCA is "+str(round(accuracy_KNN2,3)*100)+"%")
preds = KNNR.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["KNN-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[246]:


#SVM PCA


# In[252]:


svm3 = SVC(kernel = 'linear', probability = True)


# In[253]:


SVMR3 = svm3.fit(X_train2, y_train2)


# In[254]:


plot_decision_boundary(SVMR3, 'SVM-Linear', X_train2, X_test2, y_train2)


# In[256]:


predsvm3 = SVMR3.predict(X_test2)
resultsvm3 = confusion_matrix(y_true=y_test2, y_pred=predsvm3, labels=[0,1])
accuracy_svm3 = np.sum(np.diagonal(resultsvm3))/np.sum(resultsvm3)
print("SVM - Linear PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (svm3, metrics.classification_report(y_test2, predsvm3)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predsvm3))
print ("The accuracy of SVM-Linear is "+str(round(accuracy_svm3,3)*100)+"%")
preds = SVMR3.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["SVM-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:


#Kernel SVM PCA


# In[278]:


svm4 = SVC(kernel = 'rbf', gamma= 0.001, probability = True)


# In[279]:


SVMR4 = svm4.fit(X_train2, y_train2)


# In[280]:


plot_decision_boundary(SVMR4, 'Kernel - SVM', X_train2, X_test2, y_train2)


# In[282]:


predsvm4 = SVMR4.predict(X_test2)
resultsvm4 = confusion_matrix(y_true=y_test2, y_pred=predsvm4, labels=[0,1])
accuracy_svm4 = np.sum(np.diagonal(resultsvm4))/np.sum(resultsvm4)
print("Kernel SVM PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (svm4, metrics.classification_report(y_test2, predsvm4)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predsvm4))
print ("The accuracy of Kernel SVM PCA is "+str(round(accuracy_svm4,3)*100)+"%")
preds = SVMR4.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Kernel SVM-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[283]:


#Neural Network PCA


# In[284]:


nnw2 = MLPClassifier(hidden_layer_sizes = (20, 10), max_iter=10_000)


# In[285]:


nnwR2 = nnw2.fit(X_train2, y_train2)


# In[286]:


plot_decision_boundary(nnwR2, 'Neural Network', X_train2, X_test2, y_train2)


# In[287]:


prednnw2 = nnwR2.predict(X_test2)
resultsnnw2 = confusion_matrix(y_true=y_test2, y_pred=prednnw2, labels=[0,1])
accuracy_nnw2 = np.sum(np.diagonal(resultsnnw2))/np.sum(resultsnnw2)
print("Neural Network PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (nnw, metrics.classification_report(y_test2, prednnw2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, prednnw2))
print ("The accuracy of Nerual Network PCA is "+str(round(accuracy_nnw2,3)*100)+"%")
preds = nnwR2.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Neural Network-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:


#CART PCA


# In[301]:


CART2 = tree.DecisionTreeClassifier(random_state = 1)


# In[302]:


CART3 = CART2.fit(X_train2, y_train2)


# In[303]:


plot_decision_boundary(CART2, 'CART', X_train2, X_test2, y_train2)


# In[310]:


predictionCART2 = CART3.predict(X_test2)
resultCART2 = confusion_matrix(y_true=y_test2, y_pred=predictionCART2, labels=[0,1])
accuracy_CART2 = np.sum(np.diagonal(resultCART2))/np.sum(resultCART2)
print("CART PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (CART2, metrics.classification_report(y_test2, predictionCART2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predictionCART2))
print ("The accuracy of CART PCA is "+str(round(accuracy_CART2,3)*100)+"%")
preds = CART3.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["CART-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:


#Random Forest PCA


# In[306]:


RF2 = ensemble.RandomForestClassifier(random_state = 1)


# In[307]:


RF3 = RF2.fit(X_train2, y_train2)


# In[308]:


plot_decision_boundary(RF2, 'Random Forest', X_train2, X_test2, y_train2)


# In[311]:


predictionRF2 = RF3.predict(X_test2)
resultRF2 = confusion_matrix(y_true=y_test2, y_pred=predictionRF2, labels=[0,1])
accuracy_RF2 = np.sum(np.diagonal(resultRF2))/np.sum(resultRF2)
print("Random Forest PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (RF2, metrics.classification_report(y_test2, predictionRF2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predictionRF2))
print ("The accuracy of Random Forest PCA is "+str(round(accuracy_RF2,3)*100)+"%")
preds = RF3.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Random Forest-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:


#Adaboost PCA


# In[313]:


ADB2 = ensemble.AdaBoostClassifier(random_state = 1)


# In[314]:


ADB3 = ADB2.fit(X_train2, y_train2)


# In[315]:


plot_decision_boundary(ADB2, 'Ada Boost', X_train2, X_test2, y_train2)


# In[317]:


predictionADB2 = ADB3.predict(X_test2)
resultADB2 = confusion_matrix(y_true=y_test2, y_pred=predictionADB2, labels=[0,1])
accuracy_ADB2 = np.sum(np.diagonal(resultADB2))/np.sum(resultADB2)
print("AdaBoost PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (ADB2, metrics.classification_report(y_test2, predictionADB2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predictionADB2))
print ("The accuracy of AdaBoost PCA is "+str(round(accuracy_ADB2,3)*100)+"%")
preds = ADB3.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Adaboost-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:


#Gradient Boosting PCA


# In[321]:


GB2 = ensemble.GradientBoostingClassifier(random_state = 1)


# In[324]:


GB3 = GB2.fit(X_train2, y_train2)


# In[325]:


plot_decision_boundary(GB2, 'Gradient Boost', X_train2, X_test2, y_train2)


# In[326]:


predictionGB2 = GB3.predict(X_test2)
resultGB2 = confusion_matrix(y_true=y_test2, y_pred=predictionGB2, labels=[0,1])
accuracy_GB2 = np.sum(np.diagonal(resultGB2))/np.sum(resultGB2)
print("Gradient Boosting PCA:")
print("Classification report for classifier %s:\n%s\n"
      % (GB2, metrics.classification_report(y_test2, predictionGB2)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test2, predictionGB2))
print ("The accuracy of Gradient Boost PCA is "+str(round(accuracy_GB2,3)*100)+"%")
preds = GB3.predict_proba(X_test2)
for t in thresh_list:
    threshed = np.where(preds[:,1] > t, 1, 0).astype(float)
    cm = confusion_matrix(y_true=y_test, y_pred=threshed, labels=[0,1])
    accura = np.sum(np.diagonal(cm))/np.sum(cm)
    temp = pd.DataFrame({"model":["Gradient Boosting-PCA"],
                         "threshold":[t],
                         "accuracy":[accura]})
    master_tester = master_tester.append(temp)

# In[ ]:
max_record = master_tester['accuracy'].max()
max_record_selected = master_tester[master_tester['accuracy'] == max_record].copy()
print("Model with the maximum accuracy of {0}% is {1} with a threshold of {2}".format(np.round(max_record_selected.accuracy.values[0] * 100, 2), 
                                                                                     max_record_selected.model.values[0], 
                                                                                     max_record_selected.threshold.values[0]))


# In[250]:


pd.options.display.max_rows = None
master_tester


# In[62]:


# ROC Curves


# In[64]:


#SVM Final

y_score = SVMR1.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)


# In[72]:


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic SVM')
plt.legend(loc="lower right")
plt.show()


# In[80]:


#Neural Network Final
from sklearn.metrics import roc_curve, auc

y_score2 = nnw.fit(X_train, y_train).predict_proba(X_test)[:,1]



# Compute ROC curve and ROC area for each class
fpr2 = dict()
tpr2 = dict()
roc_auc2 = dict()
fpr2, tpr2, _ = roc_curve(y_test, y_score2)
roc_auc2 = auc(fpr2, tpr2)

# Compute micro-average ROC curve and ROC area
fpr2, tpr2, _ = roc_curve(y_test.ravel(), y_score2.ravel())
roc_auc2 = auc(fpr2, tpr2)


# In[81]:


plt.figure()
lw = 2
plt.plot(fpr2, tpr2, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Neural Network')
plt.legend(loc="lower right")
plt.show()


# In[84]:


#Gradient Boosting

y_score3 = GB.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr3 = dict()
tpr3 = dict()
roc_auc3 = dict()
fpr3, tpr3, _ = roc_curve(y_test, y_score3)
roc_auc3 = auc(fpr3, tpr3)

# Compute micro-average ROC curve and ROC area
fpr3, tpr3, _ = roc_curve(y_test.ravel(), y_score3.ravel())
roc_auc3 = auc(fpr3, tpr3)


# In[85]:


plt.figure()
lw = 2
plt.plot(fpr3, tpr3, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Gradient Boosting')
plt.legend(loc="lower right")
plt.show()


# In[192]:


#Logistic Regression

y_score4 = logit.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr4 = dict()
tpr4 = dict()
roc_auc4 = dict()
fpr4, tpr4, _ = roc_curve(y_test, y_score4)
roc_auc4 = auc(fpr4, tpr4)

# Compute micro-average ROC curve and ROC area
fpr4, tpr4, _ = roc_curve(y_test.ravel(), y_score4.ravel())
roc_auc4 = auc(fpr4, tpr4)


# In[193]:


plt.figure()
lw = 2
plt.plot(fpr4, tpr4, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc4)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic Logistic Regression')
plt.legend(loc="lower right")
plt.show()


# In[171]:





# In[227]:


def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)

    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()

# whatever your features are called
feature_names = ['yardline_100','yards_gained','score_differential','epa','passing_yards','rushing_yards','penalty_yards','cpoe','pass','rush','fg_made','fg_missed','is_home_team','first_down','third_down_failed','touchdown','interception','sack','tackled_for_loss','fumble']
f_importances(abs(SVMR1.coef_[0]), feature_names, top=18)


# In[303]:


clf2 =logit.fit(X_train, y_train)


# In[304]:


f_importances(abs(clf2.coef_[0]), feature_names, top=18)


# In[305]:


#Gradient Boosting - Best 
preds1 = GB.predict_proba(X_test)
threshed1 = np.where(preds1[:,1] > .4, 1, 0).astype(float)
cm1 = confusion_matrix(y_true=y_test, y_pred=threshed1, labels=[0,1])
accura1 = np.sum(np.diagonal(cm1))/np.sum(cm1)
print("Classification report for classifier %s:\n%s\n"
      % (GB, metrics.classification_report(y_test, threshed1)))
print("Confusion matrix:\n%s" % (cm1))
print ("The best accuracy of Gradient Boost is "+str(round(accura1,3)*100)+"% at threshold .4")
cm_display = ConfusionMatrixDisplay(cm1, display_labels= '01').plot()


# In[306]:


#Logistic Regression - Best
clf2 =logit.fit(X_train, y_train)
preds2 = clf2.predict_proba(X_test)
threshed2 = np.where(preds2[:,1] > .7, 1, 0).astype(float)
cm2 = confusion_matrix(y_true=y_test, y_pred=threshed2, labels=[0,1])
accura2 = np.sum(np.diagonal(cm2))/np.sum(cm2)
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, threshed2)))
print("Confusion matrix:\n%s" % (cm2))
print ("The best accuracy of Logistic Regression is "+str(round(accura2,3)*100)+"% at threshold .7")
cm_display = ConfusionMatrixDisplay(cm2, display_labels= '01').plot()


# In[307]:


#Neural Network - Best
preds3 = nnw.predict_proba(X_test)
threshed3 = np.where(preds3[:,1] > .7, 1, 0).astype(float)
cm3 = confusion_matrix(y_true=y_test, y_pred=threshed3, labels=[0,1])
accura3 = np.sum(np.diagonal(cm3))/np.sum(cm3)
print("Classification report for classifier %s:\n%s\n"
      % (nnw, metrics.classification_report(y_test, threshed3)))
print("Confusion matrix:\n%s" % (cm3))
print ("The best accuracy of Neural Network is "+str(round(accura3,3)*100)+"% at threshold .7")
cm_display = ConfusionMatrixDisplay(cm3, display_labels= '01').plot()


# In[308]:


#SVM - Best
preds4 = svm.predict_proba(X_test)
threshed4 = np.where(preds4[:,1] > .5, 1, 0).astype(float)
cm4 = confusion_matrix(y_true=y_test, y_pred=threshed4, labels=[0,1])
accura4 = np.sum(np.diagonal(cm4))/np.sum(cm4)
print("Classification report for classifier %s:\n%s\n"
      % (svm, metrics.classification_report(y_test, threshed4)))
print("Confusion matrix:\n%s" % (cm))
print ("The best accuracy of SVM is "+str(round(accura4,3)*100)+"% at threshold .5")
cm_display = ConfusionMatrixDisplay(cm4, display_labels= '01').plot()


# In[281]:


#Kernel SVM - Best
preds5 = svm2.predict_proba(X_test)
threshed5 = np.where(preds5[:,1] > .4, 1, 0).astype(float)
cm5 = confusion_matrix(y_true=y_test, y_pred=threshed5, labels=[0,1])
accura5 = np.sum(np.diagonal(cm5))/np.sum(cm5)
print("Classification report for classifier %s:\n%s\n"
      % (svm2, metrics.classification_report(y_test, threshed5)))
print("Confusion matrix:\n%s" % (cm5))
print ("The best accuracy of Kernel SVM is "+str(round(accura5,3)*100)+"% at threshold .4")


# In[284]:


#Naive Bayes - Best
NB2 = NB.fit(X_train, y_train)
preds6 = NB2.predict_proba(X_test)
threshed6 = np.where(preds6[:,1] > .9, 1, 0).astype(float)
cm6 = confusion_matrix(y_true=y_test, y_pred=threshed6, labels=[0,1])
accura6 = np.sum(np.diagonal(cm6))/np.sum(cm6)
print("Classification report for classifier %s:\n%s\n"
      % (NB, metrics.classification_report(y_test, threshed6)))
print("Confusion matrix:\n%s" % (cm6))
print ("The best accuracy of Naive Bayes is "+str(round(accura6,3)*100)+"% at threshold .9")


# In[286]:


#KNN - Best
preds7 = KNN.predict_proba(X_test)
threshed7 = np.where(preds7[:,1] > .1, 1, 0).astype(float)
cm7 = confusion_matrix(y_true=y_test, y_pred=threshed7, labels=[0,1])
accura7 = np.sum(np.diagonal(cm7))/np.sum(cm7)
print("Classification report for classifier %s:\n%s\n"
      % (KNN, metrics.classification_report(y_test, threshed7)))
print("Confusion matrix:\n%s" % (cm7))
print ("The best accuracy of KNN is "+str(round(accura7,3)*100)+"% at threshold .1")


# In[288]:


#CART - Best
preds8 = CART.predict_proba(X_test)
threshed8 = np.where(preds8[:,1] > .1, 1, 0).astype(float)
cm8 = confusion_matrix(y_true=y_test, y_pred=threshed8, labels=[0,1])
accura8 = np.sum(np.diagonal(cm8))/np.sum(cm8)
print("Classification report for classifier %s:\n%s\n"
      % (CART, metrics.classification_report(y_test, threshed8)))
print("Confusion matrix:\n%s" % (cm8))
print ("The best accuracy of CART is "+str(round(accura8,3)*100)+"% at threshold .1")


# In[289]:


#Random Forest - Best
preds9 = RF.predict_proba(X_test)
threshed9 = np.where(preds9[:,1] > .5, 1, 0).astype(float)
cm9 = confusion_matrix(y_true=y_test, y_pred=threshed9, labels=[0,1])
accura9 = np.sum(np.diagonal(cm9))/np.sum(cm9)
print("Classification report for classifier %s:\n%s\n"
      % (RF, metrics.classification_report(y_test, threshed9)))
print("Confusion matrix:\n%s" % (cm))
print ("The best accuracy of Random Forest is "+str(round(accura9,3)*100)+"% at threshold .5")


# In[290]:


#AdaBoost - Best
preds10 = ADB.predict_proba(X_test)
threshed10 = np.where(preds10[:,1] > .5, 1, 0).astype(float)
cm10 = confusion_matrix(y_true=y_test, y_pred=threshed10, labels=[0,1])
accura10 = np.sum(np.diagonal(cm10))/np.sum(cm10)
print("Classification report for classifier %s:\n%s\n"
      % (ADB, metrics.classification_report(y_test, threshed10)))
print("Confusion matrix:\n%s" % (cm))
print ("The best accuracy of AdaBoost is "+str(round(accura10,3)*100)+"% at threshold .5")


# In[292]:


#Naive Bayes PCA - Best
NB3 = NB.fit(X_train2, y_train2)
preds11 = NB3.predict_proba(X_test2)
threshed11 = np.where(preds11[:,1] > .5, 1, 0).astype(float)
cm11 = confusion_matrix(y_true=y_test, y_pred=threshed11, labels=[0,1])
accura11 = np.sum(np.diagonal(cm11))/np.sum(cm11)
print("Classification report for classifier %s:\n%s\n"
      % (NB, metrics.classification_report(y_test, threshed11)))
print("Confusion matrix:\n%s" % (cm))
print ("The best accuracy of Naive Bayes PCA is "+str(round(accura11,3)*100)+"% at threshold .5")


# In[293]:


#Logistic Regression PCA - Best
LR2 =logit.fit(X_train2, y_train2)
preds12 = LR2.predict_proba(X_test2)
threshed12 = np.where(preds12[:,1] > .5, 1, 0).astype(float)
cm12 = confusion_matrix(y_true=y_test, y_pred=threshed12, labels=[0,1])
accura12 = np.sum(np.diagonal(cm12))/np.sum(cm12)
print("Classification report for classifier %s:\n%s\n"
      % (LR2, metrics.classification_report(y_test, threshed12)))
print("Confusion matrix:\n%s" % (cm12))
print ("The best accuracy of Logistic Regression PCA is "+str(round(accura12,3)*100)+"% at threshold .5")


# In[294]:


#KNN PCA - Best
preds13 = KNNR.predict_proba(X_test2)
threshed13 = np.where(preds13[:,1] > .1, 1, 0).astype(float)
cm13 = confusion_matrix(y_true=y_test, y_pred=threshed13, labels=[0,1])
accura13 = np.sum(np.diagonal(cm13))/np.sum(cm13)
print("Classification report for classifier %s:\n%s\n"
      % (KNNR, metrics.classification_report(y_test, threshed13)))
print("Confusion matrix:\n%s" % (cm13))
print ("The best accuracy of KNN PCA is "+str(round(accura13,3)*100)+"% at threshold .1")


# In[295]:


#SVM PCA - Best
preds14 = SVMR3.predict_proba(X_test2)
threshed14 = np.where(preds14[:,1] > .5, 1, 0).astype(float)
cm14 = confusion_matrix(y_true=y_test, y_pred=threshed14, labels=[0,1])
accura14 = np.sum(np.diagonal(cm14))/np.sum(cm14)
print("Classification report for classifier %s:\n%s\n"
      % (SVMR3, metrics.classification_report(y_test, threshed14)))
print("Confusion matrix:\n%s" % (cm14))
print ("The best accuracy of SVM PCA is "+str(round(accura14,3)*100)+"% at threshold .5")


# In[296]:


#Kernel SVM PCA - Best
preds15 = SVMR4.predict_proba(X_test2)
threshed15 = np.where(preds15[:,1] > .5, 1, 0).astype(float)
cm15 = confusion_matrix(y_true=y_test, y_pred=threshed15, labels=[0,1])
accura15 = np.sum(np.diagonal(cm15))/np.sum(cm15)
print("Classification report for classifier %s:\n%s\n"
      % (SVMR4, metrics.classification_report(y_test, threshed15)))
print("Confusion matrix:\n%s" % (cm15))
print ("The best accuracy of Kernel SVM is "+str(round(accura15,3)*100)+"% at threshold .5")


# In[297]:


#Neural Network PCA - Best
preds16 = nnwR2.predict_proba(X_test2)
threshed16 = np.where(preds16[:,1] > .6, 1, 0).astype(float)
cm16 = confusion_matrix(y_true=y_test, y_pred=threshed16, labels=[0,1])
accura16 = np.sum(np.diagonal(cm16))/np.sum(cm16)
print("Classification report for classifier %s:\n%s\n"
      % (nnwR2, metrics.classification_report(y_test, threshed16)))
print("Confusion matrix:\n%s" % (cm16))
print ("The best accuracy of Neural Network PCA is "+str(round(accura16,3)*100)+"% at threshold .6")


# In[298]:


#CART PCA - Best
preds17 = CART3.predict_proba(X_test2)
threshed17 = np.where(preds17[:,1] > .1, 1, 0).astype(float)
cm17 = confusion_matrix(y_true=y_test, y_pred=threshed17, labels=[0,1])
accura17 = np.sum(np.diagonal(cm17))/np.sum(cm17)
print("Classification report for classifier %s:\n%s\n"
      % (CART3, metrics.classification_report(y_test, threshed17)))
print("Confusion matrix:\n%s" % (cm17))
print ("The best accuracy of CART PCA is "+str(round(accura17,3)*100)+"% at threshold .1")


# In[299]:


#Random Forest PCA - Best
preds18 = RF3.predict_proba(X_test2)
threshed18 = np.where(preds18[:,1] > .5, 1, 0).astype(float)
cm18 = confusion_matrix(y_true=y_test, y_pred=threshed18, labels=[0,1])
accura18 = np.sum(np.diagonal(cm18))/np.sum(cm18)
print("Classification report for classifier %s:\n%s\n"
      % (RF3, metrics.classification_report(y_test, threshed18)))
print("Confusion matrix:\n%s" % (cm18))
print ("The best accuracy of Random Forest PCA is "+str(round(accura18,3)*100)+"% at threshold .5")


# In[300]:


#AdaBoost PCA - Best
preds19 = ADB3.predict_proba(X_test2)
threshed19 = np.where(preds19[:,1] > .5, 1, 0).astype(float)
cm19 = confusion_matrix(y_true=y_test, y_pred=threshed19, labels=[0,1])
accura19 = np.sum(np.diagonal(cm19))/np.sum(cm19)
print("Classification report for classifier %s:\n%s\n"
      % (ADB3, metrics.classification_report(y_test, threshed19)))
print("Confusion matrix:\n%s" % (cm19))
print ("The best accuracy of AdaBoost PCA is "+str(round(accura19,3)*100)+"% at threshold .5")


# In[301]:


#Gradient Boosting PCA - Best
preds20 = GB3.predict_proba(X_test2)
threshed20 = np.where(preds20[:,1] > .6, 1, 0).astype(float)
cm20 = confusion_matrix(y_true=y_test, y_pred=threshed20, labels=[0,1])
accura20 = np.sum(np.diagonal(cm20))/np.sum(cm20)
print("Classification report for classifier %s:\n%s\n"
      % (GB3, metrics.classification_report(y_test, threshed20)))
print("Confusion matrix:\n%s" % (cm20))
print ("The best accuracy of Gradient Boost PCA is "+str(round(accura20,3)*100)+"% at threshold .6")


# In[ ]:




