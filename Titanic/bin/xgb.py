#import tensorflow as tf
#import multiprocessing as mp; mp.set_start_method('forkserver')
import xgboost as xgb
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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pylab as plt
from sklearn.model_selection import GridSearchCV


#from ensemble import base_train, base_test, X_train, X_test, Y_train, id_test
import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test

# hyper parameter settings
learning_rate = 0.1
gamma = 0.
seed = 42
nthread = 88
n_estimators = 1000
subsample = 0.8
colsample_bytree = 0.8

# tuned
max_depth = 3
min_child_weight = 5

# basic settings
#target = 'Survived'
#IDcols = ''


def modelfit(alg, fea_train, label_train, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        # fea_train is sparse matrix
        xgtrain = xgb.DMatrix(fea_train, label=label_train.values)
        cvresult = xgb.cv(xgb_param,
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds,
                          metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=True)

        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(fea_train, label_train, eval_metric='auc')

    #Predict training set:
    dtrain_predictions = alg.predict(fea_train)
    dtrain_predprob = alg.predict_proba(fea_train)[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % accuracy_score(label_train.values, dtrain_predictions))
    print("AUC Score (Train): %f" % roc_auc_score(label_train, dtrain_predprob))

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


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
encoded = enc.fit_transform(whole_df)
X_train = encoded[:train_rows, :]
X_test = encoded[train_rows:, :]



# tune max_depth & min_child_weight
param_t1 = {
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5]
}

#        seed=seed,
gs1 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=seed,
        nthread=88),
    param_grid=param_t1,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs1.fit(X_train.todense(), Y_train.Survived)
print(gs1.grid_scores_)
print(gs1.best_params_)
print(gs1.best_score_)

# tune parameters more accuracy

# {'max_depth': 3, 'min_child_weight': 1}
param_t2 = {
    'max_depth': [2,3,4],
    'min_child_weight': [1, 2]
}

gs2 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=seed,
        nthread=88),
    param_grid=param_t2,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs2.fit(X_train.todense(), Y_train.Survived)
print(gs2.grid_scores_)
print(gs2.best_params_)
print(gs2.best_score_)

# tuned result
# {'max_depth': 3, 'min_child_weight': 1}


# begin to tune gamma
param_t3 = {
    'gamma': [i/10.0 for i in range(0, 5)]
}

gs3 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=seed,
        max_depth=3,
        min_child_weight=1,
        nthread=88),
    param_grid=param_t3,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs3.fit(X_train.todense(), Y_train.Survived)
print(gs3.grid_scores_)
print(gs3.best_params_)
print(gs3.best_score_)

# tuned gamma
# {'gamma': 0.1}
# 0.873509384063



# begin to tune subsample & colsample_bytree

param_t4 = {
    'subsample':[i/10.0 for i in range(6, 10)],
    'colsample_bytree': [i/10.0 for i in range(6,10)]
}

gs4 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        seed=seed,
        max_depth=3,
        min_child_weight=1,
        gamma=0.1,
        nthread=88),
    param_grid=param_t4,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs4.fit(X_train.todense(), Y_train.Survived)
print(gs4.grid_scores_)
print(gs4.best_params_)
print(gs4.best_score_)
# tuned result
# {'colsample_bytree': 0.7, 'subsample': 0.9}



# for more accuracy
param_t5 = {
    'subsample':[i/100.0 for i in range(65, 80, 5)],
    'colsample_bytree': [i/100.0 for i in range(85,100, 5)]
}


gs5 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        seed=seed,
        max_depth=3,
        min_child_weight=1,
        gamma=0.1,
        nthread=88),
    param_grid=param_t5,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs5.fit(X_train.todense(), Y_train.Survived)
print(gs5.grid_scores_)
print(gs5.best_params_)
print(gs5.best_score_)
# tune result , score 有所下降
# {'colsample_bytree': 0.85, 'subsample': 0.65}

param_t6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}


gs6 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        seed=seed,
        subsample=0.65,
        colsample_bytree=0.85,
        max_depth=3,
        min_child_weight=1,
        gamma=0.1,
        nthread=88),
    param_grid=param_t6,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs6.fit(X_train.todense(), Y_train.Survived)
print(gs6.grid_scores_)
print(gs6.best_params_)
print(gs6.best_score_)
# tuned
# {'reg_alpha': 1}


param_t7 = {
    'reg_alpha':[0.5, 1, 1.5, 2, 2.5]
}


gs7 = GridSearchCV(
    estimator=xgb.XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=150,
        seed=seed,
        subsample=0.75,
        colsample_bytree=0.85,
        max_depth=3,
        min_child_weight=1,
        gamma=0.1,
        nthread=88),
    param_grid=param_t7,
    scoring='roc_auc',
    iid=False,
    cv=5)

gs7.fit(X_train.todense(), Y_train.Survived)
print(gs7.grid_scores_)
print(gs7.best_params_)
print(gs7.best_score_)
# for more accuracy
# {'reg_alpha': 2.5}



# reduce learning rate
# from 0.1 to 0.01
model = xgb.XGBClassifier(
    learning_rate=0.01,
    n_estimators=1000,
    seed=seed,
    subsample=0.75,
    colsample_bytree=0.85,
    max_depth=3,
    min_child_weight=1,
    gamma=0.1,
    reg_alpha=2.5,
    nthread=88)

#     learning_rate=learning_rate,
#     nthread=nthread,
#     gamma=gamma,
#     seed=seed,
#     max_depth=max_depth,
#     n_estimators=n_estimators,
#     subsample=subsample,
#     colsample_bytree=colsample_bytree
# )


#modelfit(model, X_train, Y_train.Survived)


model.fit(X_train, Y_train.Survived)

pred = model.predict(X_test)
result_list = [round(value) for value in pred]


result_df = pd.concat([id_test, pd.DataFrame(result_list)], axis=1)
result_df.columns = ["PassengerId" , "Survived"]
result_df['Survived'] = result_df.Survived.apply(int)
result_df.to_csv('../data/xgb_result_to_submission', index=False)
