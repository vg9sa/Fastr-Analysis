#!/usr/bin/env python
# coding: utf-8

# In[108]:


#NFL data Analysis
#ISYE 6740 Project

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns


# In[130]:


pd.options.display.max_columns = 100
pd.options.display.max_rows = 100


# In[143]:


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
                                                                                                'yards_gained': 'mean',
                                                                                                'score_differential': 'mean',
                                                                                                'epa': 'mean',
                                                                                                'passing_yards': 'mean',
                                                                                                'rushing_yards': 'mean',
                                                                                                'penalty_yards': 'mean',
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
    if year!=2020:
        combined_data_GM.append(featureData2)
    else:
        testingData_GM.append(featureData2)
    
    #Reprocess for Season Level
    featureData2.drop(['year','week','game_id'], axis = 1)
    featureData3 = featureData2.groupby(['posteam'], as_index = False).mean()
    
    #Dump Season Level Data
    if year!=2020:
        combined_data_SE.append(featureData3)
    else:
        testingData_SE.append(featureData3)

#Game level Aggregation
finalData_GM = pd.concat(combined_data_GM).reset_index(drop=True)
testingFinalData_GM = pd.concat(testingData_GM)

#Season Level Aggregation
finalData_SE = pd.concat(combined_data_SE).reset_index(drop=True)
testingFinalData_SE = pd.concat(testingData_SE)


# In[153]:


finalData_GM


# In[154]:


finalData_SE


# In[155]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


X_train,y_train = finalData_SE.iloc[:,2:21],finalData_SE.iloc[:,22]
X_test, y_test = testingFinalData_SE.iloc[:,2:21], testingFinalData_SE.iloc[:,22]

clf = LogisticRegression(max_iter = 1000).fit(X_train, y_train)
prediction = clf.predict(X_test)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, prediction)
print("Precision \n", precision)
print("\nRecall \n", recall)
print("\nF-score \n", fscore)
logistic_confusion = confusion_matrix(y_test,prediction)
print(str(logistic_confusion)) 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




