#NFL data Analysis
#ISYE 6740 Project

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import seaborn as sns


#loading data
years = [2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006]
#years = [2020,2019,2018,2017]
combined_data = []
testingData = []
for year in years:
    data = pd.read_csv('https://github.com/guga31bb/nflfastR-data/blob/master/data/' \
                             'play_by_play_' + str(year) + '.csv.gz?raw=True',
                             compression='gzip', low_memory=False)
    data.drop(['passer_player_name', 'passer_player_id',
               'rusher_player_name', 'rusher_player_id',
               'receiver_player_name', 'receiver_player_id'],
              axis=1, inplace=True)
    data = data.loc[(data.play_type.isin(['no_play','pass','run'])) & (data.epa.isna()==False)]
    data.play_type.loc[data['pass']==1] = 'pass'
    data.play_type.loc[data.rush==1] = 'run'
    data.reset_index(drop=True, inplace=True)
    nflData = data.copy()
    playoffData = nflData[nflData['season_type']=='POST']
    regSznData = nflData[nflData['season_type']=='REG']
    playoff_teams = np.unique(playoffData[['home_team','away_team']].values).tolist()
    allTeams = np.unique(regSznData[['home_team','away_team']].values).tolist()

    features = ['posteam','yardline_100','yards_gained','score_differential','epa','passing_yards','rushing_yards','penalty_yards','cpoe']
    featureData = regSznData[features]
    featureData = featureData.groupby(['posteam'], as_index = False).mean()

    def playoffCheck(x):
        if x in playoff_teams:
            return 1
        else:
            return 0

    featureData['playoffs'] = featureData.apply(lambda row: playoffCheck(row['posteam']),axis=1)
    if year!=2020:
        combined_data.append(featureData)
    else:
        testingData.append(featureData)
finalData = pd.concat(combined_data).reset_index(drop=True)
testingFinalData = pd.concat(testingData)
print(testingFinalData)
X_train,y_train = finalData.iloc[:,1:8],finalData.iloc[:,9]
X_test, y_test = testingFinalData.iloc[:,1:8], testingFinalData.iloc[:,9]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

clf = LogisticRegression(max_iter = 1000).fit(X_train, y_train)
prediction = clf.predict(X_test)
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, prediction)
print("Precision \n", precision)
print("\nRecall \n", recall)
print("\nF-score \n", fscore)
logistic_confusion = confusion_matrix(y_test,prediction)
print(str(logistic_confusion))
