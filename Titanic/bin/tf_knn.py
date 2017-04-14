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
encoded = enc.fit_transform(whole_df)
X_train = encoded[:train_rows, :]
X_test = encoded[train_rows:, :]


# feature selection
train = whole_df[:train_rows]
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(X_train, Y_train.Survived)
# find out feature importance
features = pd.DataFrame()
# after one-hot encoded not care the columns name
#features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'],ascending=False)

# # only using top n features
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test =  model.transform(X_test)


X_train= X_train.toarray()
X_test = X_test.toarray()


sample_size, fea_num = X_train.shape
label_num = Y_train.shape[1]
point_num = X_test.shape[0]

X = tf.placeholder(tf.float32, [None, fea_num])
new_point = tf.placeholder(tf.float32, [fea_num])
Y = tf.placeholder(tf.float32, [None, label_num])

# L1 distance
distance = tf.reduce_sum(tf.abs(X - new_point), reduction_indices=1)

# L2 distance
distance2 = tf.sqrt(tf.reduce_sum(tf.square(X - new_point), reduction_indices=1))

pred = tf.argmin(distance2, 0)

accurancy = 0.


init = tf.global_variables_initializer()

result_list = []
# Launch  graph
with tf.Session() as sess:
        sess.run(init)

        # calc every point class
        for i in range(point_num):
            # get nearest neighbor, using all training data
            knn_index = sess.run(pred,
                                 feed_dict={
                                     X: X_train,
                                     new_point: X_test[i]})



            result_list.append(Y_train.Survived[knn_index ])

        result_df = pd.concat([id_test, pd.DataFrame(result_list)], axis=1)
        result_df.columns = ["PassengerId" , "Survived"]
        result_df.to_csv('../data/knn_result_to_submission', index=False)


            #print("knn_index:", knn_index)
            #print("nearest class is:", Y_train.iloc[knn_index, ].as_matrix())
            #print("np.argmax(Y_train[knn_index]):", np.argmax(Y_train.iloc[knn_index, ]))

        #     print "Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
        #         "True Class:", np.argmax(Yte[i])
        #     # 计算 accuracy
        #     if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
        #         accuracy += 1. / len(Xte)
        # print("Done!")
        # print("Accuracy:", accuracy)





