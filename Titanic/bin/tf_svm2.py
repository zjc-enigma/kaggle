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
import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test

Y_train = Y_train.replace(0, -1) # for svm training

# one-hot encoding
enc = OneHotEncoder()
whole_df = pd.concat([X_train, X_test], axis=0)
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
# train = whole_df[:train_rows]
clf = ExtraTreesClassifier(n_estimators=200)
clf = clf.fit(X_train, Y_train.Survived)
# find out feature importance
features = pd.DataFrame()
# after one-hot encoded not care the columns name
#features['feature'] = X_train.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'],ascending=False)

# only using top n features
model = SelectFromModel(clf, prefit=True)
X_train = model.transform(X_train)
X_test =  model.transform(X_test)

# parameter setting
SVMC = 1
GAMMA = 50.
EPOCH_NUM = 2000
BATCH_SIZE = 200
LEARNING_RATE = 0.05
DISPLAY_STEP = 2


sample_size, feature_num = X_train.shape
label_num = Y_train.shape[1]


X = tf.placeholder(tf.float32, [None, feature_num])
Y = tf.placeholder(tf.float32, [None, label_num])
test = tf.placeholder(tf.float32, [None, feature_num])
beta = tf.Variable(tf.truncated_normal(shape=[1, BATCH_SIZE], stddev=.1))

gamma = tf.constant(-GAMMA)

def Kernel_Train():
    tmp_abs = tf.reshape(tensor=tf.reduce_sum(tf.square(X), axis=1), shape=[-1,1])
    tmp_ = tf.add(tf.subtract(tmp_abs, 2.*tf.matmul(X, tf.transpose(X))), tf.transpose(tmp_abs))
    return tf.exp(gamma*tf.abs(tmp_))


def cost_func():
    left = tf.reduce_sum(beta)
    beta_square = tf.matmul(beta, beta, transpose_a=True)
    Y_square = tf.matmul(Y, Y, transpose_b= True)
    right = tf.reduce_sum(Kernel_Train()*beta_square*Y_square)
    return -tf.subtract(left, right)


def Kernel_Prediction():
    tmpA = tf.reshape(tf.reduce_sum(tf.square(X), 1),[-1,1])
    tmpB = tf.reshape(tf.reduce_sum(tf.square(test), 1),[-1,1])
    tmp = tf.add(tf.subtract(tmpA, 2.*tf.matmul(X, test, transpose_b=True)), tf.transpose(tmpB))
    return tf.exp(gamma*tf.abs(tmp))


def Prediction():
    kernel_out = tf.matmul(tf.transpose(Y)*beta, Kernel_Prediction())
    return tf.sign(kernel_out - tf.reduce_mean(kernel_out))


cost = cost_func()
pred = Prediction()
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(EPOCH_NUM):
        avg_cost = 0.
        total_batch = int(sample_size/BATCH_SIZE) # for the rest points
        #total_batch = int(sample_size/BATCH_SIZE)

        for i in range(total_batch):
            begin_idx = BATCH_SIZE*i
            #print('begin idx:', begin_idx, '| end idx:', begin_idx+BATCH_SIZE)
            batch_x = X_train[begin_idx:begin_idx+BATCH_SIZE]
            batch_y = Y_train[begin_idx:begin_idx+BATCH_SIZE]
            _, c = sess.run([optimizer, cost],
                            feed_dict={
                                X: batch_x.toarray(),
                                Y: batch_y.as_matrix()
                            })
            avg_cost += c / total_batch

        # cover the rest training point
        # begin_idx = BATCH_SIZE*total_batch
        # batch_x = X_train[begin_idx:]
        # batch_y = Y_train[begin_idx:]
        # _, c = sess.run([optimizer, cost],
        #                 feed_dict={
        #                     #X: batch_x.as_matrix(),
        #                     X: batch_x.toarray(),
        #                     Y: batch_y.as_matrix()
        #                 })
        # avg_cost += c / total_batch


        if (epoch+1) % DISPLAY_STEP == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "avg Cost={}".format(avg_cost))



    output = sess.run(pred, feed_dict={X:X_train, Y:Y_train, test:test})
