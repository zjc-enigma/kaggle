import pandas as pd
import xgboost as xgb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test


svm_result = pd.read_csv("../data/svm_result_to_submission")
lr_result = pd.read_csv("../data/result_to_submission")
nn_result = pd.read_csv("../data/nn_result_to_submission")
knn_result = pd.read_csv("../data/knn_result_to_submission")
xgb_result = pd.read_csv("../data/xgb_result_to_submission")

all_df = pd.merge(svm_result, lr_result, on='PassengerId')
all_df = pd.merge(all_df, nn_result, on='PassengerId')
all_df = pd.merge(all_df, knn_result, on='PassengerId')
all_df = pd.merge(all_df, xgb_result, on='PassengerId')


del all_df['PassengerId']
all_df.columns = ['svm', 'lr', 'nn', 'knn', 'xgb']

# hyper parameter settings
learning_rate = 0.1
gamma = 0.
seed = 42
nthread = 88
max_depth = 5
n_estimators = 1000
subsample = 0.8
colsample_bytree = 0.8


model = xgb.XGBClassifier(
    learning_rate=learning_rate,
    nthread=nthread,
    gamma=gamma,
    seed=seed,
    max_depth=max_depth,
    n_estimators=n_estimators,
    subsample=subsample,
    colsample_bytree=colsample_bytree)


pred = model.fit(all_df, Y_train.Survived)
result_list = [round(value) for value in pred]
result_df = pd.concat([id_test, pd.DataFrame(result_list)], axis=1)
result_df.columns = ["PassengerId" , "Survived"]
#result_df['Survived'] = result_df.Survived.apply(int)
result_df.to_csv('../data/xgb_result_to_submission', index=False)




