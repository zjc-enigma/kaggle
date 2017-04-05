import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test

# parameters
learning_rate = 0.01
training_epochs = 10000
batch_size = 500
display_step = 1
kfold_split_num = 8
gpu_num = 2

# make CV
# kfold = KFold(n_splits=kfold_split_num)
# kf = kfold.split(X_train, Y_train)
# cv_list = []
# for train_idx, validation_idx in kf:
#     cv_list.append((train_idx, validation_idx))
#Y_train = pd.get_dummies(Y_train)
# input
# sample_size = len(cv_list[0][0])

sample_size, fea_num = X_train.shape
label_num = Y_train.shape[1]

#for d in ['/gpu:0', '/gpu:1']:
with tf.device('/cpu:0'):
    X = tf.placeholder(tf.float32, [None, fea_num])
    Y = tf.placeholder(tf.float32, [None, label_num])
    W = tf.Variable(tf.zeros([fea_num, label_num]))
    b = tf.Variable(tf.zeros([label_num]))

# logistic regression
with tf.device('/gpu:0'):
    logit_func = tf.matmul(X, W) + b
    pred = tf.nn.sigmoid(logit_func)
    Y_pred = tf.cast(pred > 0.5, tf.float32)
    accurancy = tf.reduce_mean(tf.cast(tf.equal(Y_pred, Y), tf.float32))

with tf.device('/cpu:0'):
    auc = tf.contrib.metrics.streaming_auc(Y, Y_pred)

# cross entropy cost
#cost_func = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), 1))
#cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
with tf.device('/gpu:1'):
    cost_func = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logit_func,
            labels=Y))

#cost_func = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
#cost_func = tf.reduce_mean(tf.reduce_sum((-Y*tf.log(pred))-((1-Y)*tf.log(1-pred)), 1))

with tf.device('/gpu:1'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

init = tf.global_variables_initializer()
# run
#with tf.Session() as sess:
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

    sess.run(init)
    sess.run(tf.local_variables_initializer())
    #accurancy_list = []
    #for cv_num, (train_idx, validation_idx) in enumerate(cv_list):

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(sample_size/batch_size) + 1 # for the rest points
        #total_batch = int(sample_size/batch_size)

        for i in range(total_batch):
            begin_idx = batch_size*i
            #print('begin idx:', begin_idx, '| end idx:', begin_idx+batch_size)
            batch_x = X_train[begin_idx:begin_idx+batch_size]
            batch_y = Y_train[begin_idx:begin_idx+batch_size]
            _, c = sess.run([optimizer, cost_func],
                            feed_dict={
                                X: batch_x.as_matrix(),
                                Y: batch_y.as_matrix()
                            })
            avg_cost += c / total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "avg Cost={}".format(avg_cost),
                  "W={}".format(sess.run(W)))
                  # "cv={}".formate(cv_num))
                  # "b={}".format(str(sess.run(b))))

    #print(X_test.as_matrix()*sess.run(W) + sess.run(b))

    # accurancy_validate = sess.run(accurancy,
    #                               feed_dict={
    #                                   X: X_train.ix[validation_idx].as_matrix(),
    #                                   Y: Y_train.ix[validation_idx].as_matrix(),
    #                               })

    # accurancy_list.append(accurancy_validate)

    # print("accurancy on validation set:", accurancy_validate)
    # auc_validate = sess.run(auc,
    #                         feed_dict={
    #                             X: X_train.ix[validation_idx].as_matrix(),
    #                             Y: Y_train.ix[validation_idx].as_matrix(),
    #                         })
    #print("auc on validation set:", auc_validate)

    print("Optimization Finished!")
    #avg_accurancy = tf.reduce_mean(accurancy_list)
    #print('-----------avg accurancy is:', sess.run(avg_accurancy))

    pred_result = sess.run(pred,
                           feed_dict={
                               X: X_test.as_matrix()
                           })

    #result_df =  pd.concat([id_test, pd.DataFrame(pred_result).ix[:,1]], axis=1)
    result_df =  pd.concat([id_test, pd.DataFrame(pred_result)], axis=1)
    result_df.columns = ["PassengerId" , "Survived"]
    result_df.loc[(result_df['Survived'] >= 0.5), 'Survived'] = 1
    result_df.loc[(result_df['Survived'] < 0.5), 'Survived'] = 0
    result_df['Survived'] = result_df.Survived.apply(int)
    result_df.to_csv('../data/result_to_submission', index=False)



