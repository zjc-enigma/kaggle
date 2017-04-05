import tensorflow as tf
import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder


import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test, id_test

# parameters
learning_rate = 0.05
training_epochs = 5000
batch_size = 200
display_step = 1
kfold_split_num = 8
gpu_num = 2
regularizer_beta = 0.01

# make CV
# kfold = KFold(n_splits=kfold_split_num)
# kf = kfold.split(X_train, Y_train)
# cv_list = []
# for train_idx, validation_idx in kf:
#     cv_list.append((train_idx, validation_idx))
#Y_train = pd.get_dummies(Y_train)
# input
# sample_size = len(cv_list[0][0])

# one-hot encoding
enc = OneHotEncoder()
whole_df = pd.concat([X_train, X_test], axis=0)
encoded = enc.fit_transform(whole_df)
train_rows = X_train.shape[0]
X_train = encoded[:train_rows, :]
X_test = encoded[train_rows:, :]
#X_csr = enc.fit_transform(X_train)
#Y_csr = enc.fit_transform(Y_train)

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)



# split train / validate set
X_train, X_validate, Y_train, Y_validate = train_test_split(X_train,
                                                            Y_train,
                                                            test_size=0.3,
                                                            random_state=42)



#X_train = convert_sparse_matrix_to_sparse_tensor(X_csr)
sample_size, fea_num = X_train.shape
label_num = Y_train.shape[1]

X = tf.placeholder(tf.float32, [None, fea_num])
Y = tf.placeholder(tf.float32, [None, label_num])
W = tf.Variable(tf.zeros([fea_num, label_num]))
b = tf.Variable(tf.zeros([label_num]))

logit_func = tf.matmul(X, W) + b
pred = tf.nn.sigmoid(logit_func)
Y_pred = tf.cast(pred > 0.5, tf.float32)
accurancy = tf.reduce_mean(tf.cast(tf.equal(Y_pred, Y), tf.float32))

auc = tf.contrib.metrics.streaming_auc(Y, Y_pred)

# cross entropy cost
#cost_func = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), 1))
#cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))

loss =  tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_func, labels=Y)
regularizer = tf.nn.l2_loss(W)

cost_func = tf.reduce_mean(loss + regularizer_beta * regularizer)
#    + regular_lambda * tf.matmul(tf.transpose(W), W) # L2 regularization
#cost_func = -tf.reduce_sum(Y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))
#cost_func = tf.reduce_mean(tf.reduce_sum((-Y*tf.log(pred))-((1-Y)*tf.log(1-pred)), 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

init = tf.global_variables_initializer()
# run
with tf.Session() as sess:
#with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    #accurancy_list = []
    #for cv_num, (train_idx, validation_idx) in enumerate(cv_list):

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(sample_size/batch_size) # for the rest points
        #total_batch = int(sample_size/batch_size)

        for i in range(total_batch):
            begin_idx = batch_size*i
            #print('begin idx:', begin_idx, '| end idx:', begin_idx+batch_size)
            batch_x = X_train[begin_idx:begin_idx+batch_size]
            batch_y = Y_train[begin_idx:begin_idx+batch_size]
            _, c = sess.run([optimizer, cost_func],
                            feed_dict={
                                #X: batch_x.as_matrix(),
                                X: batch_x.toarray(),
                                Y: batch_y.as_matrix()
                            })
            avg_cost += c / total_batch

        # cover the rest training point
        begin_idx = batch_size*total_batch
        batch_x = X_train[begin_idx:]
        batch_y = Y_train[begin_idx:]
        _, c = sess.run([optimizer, cost_func],
                        feed_dict={
                            #X: batch_x.as_matrix(),
                            X: batch_x.toarray(),
                            Y: batch_y.as_matrix()
                        })
        avg_cost += c / total_batch


        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "avg Cost={}".format(avg_cost))
                  # "W={}".format(sess.run(W)))
                  # "cv={}".formate(cv_num))
                  # "b={}".format(str(sess.run(b))))

    print("Optimization Finished!")

    accurancy_validate = sess.run(accurancy,
                                  feed_dict={
                                      X: X_validate.toarray(),
                                      Y: Y_validate.as_matrix(),
                                  })

    print("accurancy on validation set:", accurancy_validate)


    auc_validate = sess.run(auc,
                            feed_dict={
                                X: X_validate.toarray(),
                                Y: Y_validate.as_matrix(),
                            })
    print("auc on validation set:", auc_validate)




    cost_validate = sess.run(cost_func,
                             feed_dict={
                                X: X_validate.toarray(),
                                Y: Y_validate.as_matrix(),
                             })
    print("cost on validation set:", cost_validate)

    pred_result = sess.run(pred,
                           feed_dict={
                               X: X_test.toarray()
                           })

    #result_df =  pd.concat([id_test, pd.DataFrame(pred_result).ix[:,1]], axis=1)
    result_df =  pd.concat([id_test, pd.DataFrame(pred_result)], axis=1)
    result_df.columns = ["PassengerId" , "Survived"]
    result_df.loc[(result_df['Survived'] >= 0.5), 'Survived'] = 1
    result_df.loc[(result_df['Survived'] < 0.5), 'Survived'] = 0
    result_df['Survived'] = result_df.Survived.apply(int)
    result_df.to_csv('../data/result_to_submission', index=False)



