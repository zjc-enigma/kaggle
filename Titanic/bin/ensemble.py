import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectFromModel
# Going to use these 5 base models for the stacking
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test

# using one-hot encoding
enc = OneHotEncoder()
whole_df = pd.concat([X_train, X_test], axis=0)
object_df = whole_df.select_dtypes(include=[object])
rest_df = whole_df.select_dtypes(exclude=[object])
lenc = LabelEncoder()
labeled_df = object_df.apply(lenc.fit_transform)
whole_df = pd.concat([rest_df, labeled_df], 1)
train_rows = X_train.shape[0]

encoded = enc.fit_transform(whole_df)
X_train = pd.DataFrame(encoded[:train_rows, :].todense())
X_test = pd.DataFrame(encoded[train_rows:, :].todense())


# whole_df = pd.concat([X_train, X_test], axis=0)
# object_df = whole_df.select_dtypes(include=[object])
# rest_df = whole_df.select_dtypes(exclude=[object])
# lenc = LabelEncoder()
# labeled_df = object_df.apply(lenc.fit_transform)
# whole_df = pd.concat([rest_df, labeled_df], 1)
# train_rows = X_train.shape[0]
# X_train = whole_df[:train_rows]
# X_test = whole_df[train_rows:]


# parameters
learning_rate = 0.05
training_epochs = 2000
batch_size = 200
display_step = 1
kfold_split_num = 8
gpu_num = 2
regularizer_beta = 0.001



# Some useful parameters which will come in handy later on
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kfold = KFold(n_splits=NFOLDS, random_state=SEED)
kf = kfold.split(X_train, Y_train)
#kf = KFold(ntrain, n_splits=NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        importances = self.clf.fit(x,y).feature_importances_
        print(importances)
        return importances
    
# Class to extend XGboost classifer



# out-of-fold
def get_oof(clf, x_train, y_train, x_test):

    kfold = KFold(n_splits=NFOLDS, random_state=SEED)
    kf = kfold.split(X_train, Y_train)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
}

# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

x_train = X_train.values
y_train = Y_train.Survived.ravel()
#y_train = Y_train.values
x_test = X_test.values

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier


rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)


cols = X_train.columns.values
# Create a dataframe with features
feature_df = pd.DataFrame( {'features': cols,
                            'rf_feature_importances': rf_feature,
                            'ef_feature_importances': et_feature,
                            'ada_feature_importances': ada_feature,
                            'gb_feature_importances': gb_feature})


feature_df['mean'] = feature_df[['rf_feature_importances',
                                 'ef_feature_importances',
                                 'ada_feature_importances',
                                 'gb_feature_importances']].mean(axis=1) # axis = 1 computes the mean row-wise
feature_df.head(3)
feature_df.sort_values(by='mean', ascending=False, inplace=True)
selected_feature_idx = feature_df.head(20).features.ravel()

X_train = X_train.iloc[: ,selected_feature_idx]
X_test = X_test.iloc[: ,selected_feature_idx]

x_train = X_train.values
x_test = X_test.values

# using new features re-train
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier


print("Training is complete")

base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                       'ExtraTrees': et_oof_train.ravel(),
                                       'AdaBoost': ada_oof_train.ravel(),
                                       'GradientBoost': gb_oof_train.ravel() })
base_predictions_train.head()


base_train = np.concatenate(( et_oof_train,
                              rf_oof_train,
                              ada_oof_train,
                              gb_oof_train,
                              svc_oof_train), axis=1)

base_test = np.concatenate(( et_oof_test,
                             rf_oof_test,
                             ada_oof_test,
                             gb_oof_test,
                             svc_oof_test), axis=1)


svm_result = pd.read_csv("../data/svm_result_to_submission")
lr_result = pd.read_csv("../data/result_to_submission")
nn_result = pd.read_csv("../data/nn_result_to_submission")
knn_result = pd.read_csv("../data/knn_result_to_submission")
xgb_result = pd.read_csv("../data/xgb_result_to_submission")

#all_df = pd.merge(svm_result, lr_result, on='PassengerId')
#all_df = pd.merge(all_df, nn_result, on='PassengerId')
#all_df = pd.merge(all_df, knn_result, on='PassengerId')
all_df = pd.merge(nn_result, lr_result, on='PassengerId')
all_df = pd.merge(all_df, xgb_result, on='PassengerId')

# all_df = pd.concat([all_df,
#                  pd.DataFrame(et_oof_test),
#                  pd.DataFrame(rf_oof_test),
#                  pd.DataFrame(ada_oof_test),
#                  pd.DataFrame(gb_oof_test),
#                  pd.DataFrame(svc_oof_test)], axis=1)



all_df['Survived_mean'] = all_df.iloc[:,1:].mean(axis=1)
all_df = all_df[['PassengerId', 'Survived_mean']]
all_df.loc[(all_df['Survived_mean'] >= 0.5), 'Survived'] = 1
all_df.loc[(all_df['Survived_mean'] < 0.5), 'Survived'] = 0
all_df = all_df[['PassengerId','Survived']]
all_df['Survived'] = all_df.Survived.apply(int)
all_df.to_csv('../data/ensemble_result_to_submission', index=False)


# one-hot encoding
# enc = OneHotEncoder()
# whole_df = pd.concat([X_train, X_test], axis=0)
# object_df = whole_df.select_dtypes(include=[object])
# rest_df = whole_df.select_dtypes(exclude=[object])
# lenc = LabelEncoder()
# labeled_df = object_df.apply(lenc.fit_transform)
# whole_df = pd.concat([rest_df, labeled_df], 1)
# train_rows = X_train.shape[0]

# encoded = enc.fit_transform(whole_df)
# X_train = encoded[:train_rows, :]
# X_test = encoded[train_rows:, :]


