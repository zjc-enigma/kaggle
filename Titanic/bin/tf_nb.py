import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from ensemble import base_train, base_test, X_train, X_test, Y_train, id_test
import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test
# one-hot encoding
enc = OneHotEncoder()
whole_df = pd.concat([X_train, X_test], axis=0)
whole_df['Age'] = whole_df.Age.apply(int)
object_df = whole_df.select_dtypes(include=[object])
rest_df = whole_df.select_dtypes(exclude=[object])
lenc = LabelEncoder()
labeled_df = object_df.apply(lenc.fit_transform)
whole_df = pd.concat([rest_df, labeled_df], 1)
train_rows = X_train.shape[0]
#encoded = enc.fit_transform(whole_df)
#encoded = whole_df
#X_train = encoded[:train_rows, :]
#X_test = encoded[train_rows:, :]
X_train = whole_df[:train_rows]
X_test = whole_df[train_rows:]

# NB: p(predict|features) = p(features, predict)/p(features)
#                         = p(features|predict)p(predict)/p(features)

all_train = pd.concat([X_train, Y_train], axis=1)
target_group = all_train.groupby('Survived')
not_survived, survived = target_group.size()
survive_rate = survived/(not_survived + survived)

fea_cols = X_train.columns.values
fea_sr = {}
# train
for col in fea_cols:
    p = target_group[col].value_counts()
    temp_df = pd.concat([p[0], p[1]], axis=1).fillna(0)
    sr = temp_df.iloc[:,1]/(temp_df.iloc[:,0]+temp_df.iloc[:,1] )
    fea_sr[col] = sr.to_dict()

# predict
for col in fea_cols:
    print('mapping column', col)
    sr = fea_sr[col]
    #X_test.replace({col:fea_sr[col]}, inplace=True)
    X_test[col] = X_test[col].map(fea_sr[col])

X_test = X_test.fillna(0.5)

survive_score = X_test.prod(axis=1)*survive_rate

not_survive_score = (1 - X_test).prod(axis=1)*(1-survive_rate)


result_df = pd.DataFrame()
result_df['PassengerId'] = id_test
result_df['Survived'] = survive_score >= not_survive_score
result_df['Survived'] = result_df.Survived.apply(int)
result_df.to_csv('../data/nb_result_to_submission', index=False)
