import json
#import _cvxcore
#import cvxpy
import fancyimpute
from fancyimpute import KNN, NuclearNormMinimization, SoftImpute, BiScaler,SimpleFill
import numpy as np
import pandas as pd
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle
from sklearn.model_selection import train_test_split
from datetime import timedelta
from datetime import datetime


mimic = pd.read_csv('../initial/labs_cohort2.csv', header=0)
head = mimic.head()     # 5 rows × 22 columns
shape = mimic.shape     # (206876, 20)
sum = (mimic.isnull().sum()).sum()      # 2815989

# 206876*20 = 4137520
# 2815989/4137520 = 0.6805982810959222
len = len(mimic['subject_id'].unique())       # 17111

# only caculate the different subject_id items and show
mimic.groupby(['subject_id']).count()       # 17111 rows × 21 columns

# show subject_id == 4 items
mimic[mimic['subject_id']==4]       # 5 rows × 22 columns

# show the number of items == null
mimic.isnull().sum()

# sort the values by subject_id and charttime
mimic = mimic.sort_values(by=['subject_id', 'charttime'])

mimic_pad = pd.concat([
        d.reset_index(drop=True).reindex(range(20)).assign(subject_id=n)
        for n, d in mimic.groupby(['subject_id'])
        ], ignore_index=True)

# padding the null values of items using 0
mimic_pad = mimic_pad.fillna(0)

rows = mimic_pad.groupby('subject_id').count()['charttime'] - 20
rows = pd.DataFrame(rows)
rows = rows.reset_index()

# show padded items
mimic_pad.head()

mimic_pad.shape     # (342220, 22)

# show padded items only subject_id == 4
mimic_pad[mimic_pad['subject_id'] == 4]     # 20 rows × 22 columns

final = pd.DataFrame()
for seq in range(rows.shape[0]):
    # print(rows.iloc[seq,0])
    # print(rows.iloc[seq,1])
    temp = mimic_pad[mimic_pad['subject_id'] == rows.iloc[seq, 0]]
    # print(temp.shape)
    nrows = temp.shape[0] - rows.iloc[seq, 1]
    temp = temp.iloc[0:nrows, :]
    # print(temp.shape)
    # print(temp.shape[0]%20)
    final = pd.concat([final, temp])

final.shape     # (342220, 22)

temp = final.iloc[:, 2:].shape
temp = final.iloc[:, 2:]

# show only WBC item
temp['WBC'].shape      # (342220,)

# number of HEMOGLOBIN == 0
(temp[temp['HEMOGLOBIN'] == 0].count())

# 342220*20
# 5607833/6844400       # 0.8193315703348723

final.groupby('subject_id').count()['charttime'] % 20
# print(maskTrain['interval'].isna().sum())
# print(activityTrain['Time'].isna().sum())
final.head()        # 5 rows × 22 columns
final = final.fillna(0)
final[final['subject_id'] == 4]     # 20 rows × 22 columns

# show all the number of subject_id, 4 ~ 99995
ids = final['subject_id'].unique()

# create the train, validation and test set
X_train, X_test = train_test_split(ids, test_size=0.20, random_state=42)
X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=52)

mimicTrain = final[final['subject_id'].isin(X_train)]
mimicVal = final[final['subject_id'].isin(X_val)]
mimicTest = final[final['subject_id'].isin(X_test)]

# create the train, validation and test mask set
# maskTrain = pd.DataFrame(np.ones((activityTrain.shape[0], activityTrain.shape[1])),columns=activityTrain.columns)
# maskTrain['Time']=activityTrain['Time']
maskTrain = mimicTrain[['subject_id', 'charttime', 'WBC']].copy()
maskTrain = maskTrain.fillna(0)
maskTrain[maskTrain['WBC'] != 0] = 1
maskTrain['charttime'] = mimicTrain['charttime'].copy()
maskTrain['subject_id'] = mimicTrain['subject_id'].copy()

maskVal = mimicVal[['subject_id', 'charttime', 'WBC']].copy()
maskVal = maskVal.fillna(0)
maskVal[maskVal['WBC'] != 0] = 1
maskVal['charttime'] = mimicVal['charttime'].copy()
maskVal['subject_id'] = mimicVal['subject_id'].copy()

maskTest = mimicTest[['subject_id', 'charttime', 'WBC']].copy()
maskTest = maskTest.fillna(0)
maskTest[maskTest['WBC'] != 0] = 1
maskTest['charttime'] = mimicTest['charttime'].copy()
maskTest['subject_id'] = mimicTest['subject_id'].copy()

maskTest.head()
maskVal.head()
maskTrain.head()


# decay: forward proccess
def decay(data):
    # print(data.head())
    data['interval'] = 0
    # df=data.groupby('person_id')
    j = 0
    for n in range(int(data.shape[0]/20)):
        i = 0
        # print(n)
        df_group = data.iloc[n*20:(n*20)+20, :]
        for index, row in df_group.iterrows():      # go over mask
            # print(row['charttime'])
            if(i == 0):
                row['interval'] = 0
                i = 1
            else:
                if(row['charttime'] == 0):
                    row['interval'] = 3600 + prev['interval']
                elif(prev['WBC'] == 1):
                    row['interval'] = timedelta.total_seconds(datetime.strptime(str(row['charttime']),"%Y-%m-%dT%H:%M:%SZ")-datetime.strptime(str(prev['charttime']),"%Y-%m-%dT%H:%M:%SZ"))
                elif(prev['WBC']==0):
                    row['interval'] =timedelta.total_seconds(datetime.strptime(str(row['charttime']),"%Y-%m-%dT%H:%M:%SZ")-datetime.strptime(str(prev['charttime']),"%Y-%m-%dT%H:%M:%SZ"))+prev['interval']
            prev = row
            data.iloc[j, 3] = row['interval']
            j = j+1

    data['interval'] = data['interval'].apply(lambda x: abs(x)/60)
    # print(data.head())
    return data


# redecay: backward proccess
def rdecay(data):
    # print(data.head())
    data['intervalReverse'] = 0
    j = data.shape[0]-1
    for n in range(int(data.shape[0]/20)):
        i = 0
        df_group = data.iloc[n*20:(n*20)+20, :]
        df_group = df_group[::-1]
        for index, row in df_group.iterrows():  # go over mask
            if(i == 0):
                row['intervalReverse'] =0
                i = 1
            else:
                if(prev['charttime'] == 0):
                    row['intervalReverse'] = 3600+prev['intervalReverse']
                elif(prev['WBC'] == 1):
                    row['intervalReverse'] = timedelta.total_seconds(datetime.strptime(str(row['charttime']), "%Y-%m-%dT%H:%M:%SZ")-datetime.strptime(str(prev['charttime']), "%Y-%m-%dT%H:%M:%SZ"))
                elif(prev['WBC'] == 0):
                    row['intervalReverse'] = timedelta.total_seconds(datetime.strptime(str(row['charttime']), "%Y-%m-%dT%H:%M:%SZ")-datetime.strptime(str(prev['charttime']), "%Y-%m-%dT%H:%M:%SZ"))+prev['interval']
            prev = row
            data.iloc[j, 4] = row['intervalReverse']
            j = j-1

    data['intervalReverse'] = data['intervalReverse'].apply(lambda x: abs(x)/60)
    # print(data.head())
    return data


# process the mask
maskTrain = decay(maskTrain)
print(maskTrain.head())
maskTrain = rdecay(maskTrain)

maskVal = decay(maskVal)
print(maskVal.head())
maskVal = rdecay(maskVal)

maskTest = decay(maskTest)
print(maskTest.head())
maskTest = rdecay(maskTest)

mimicTrain.to_csv('test/mimicTrain.csv', index=False)
maskTrain.to_csv('test/mimicTrainMask.csv', index=False)

mimicVal.to_csv('test/mimicVal.csv', index=False)
maskVal.to_csv('test/mimicValMask.csv', index=False)

mimicTest.to_csv('test/mimicTest.csv', index=False)
maskTest.to_csv('test/mimicTestMask.csv', index=False)

print(mimicTest.head())
print(maskTest.head())
























