import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import pdb
sys.path.append('../lib')
from data import X_train, Y_train, X_test

# Parameters
learning_rate = 0.01
training_epochs = 300
batch_size = 200
display_step = 1

Y_train = pd.get_dummies(Y_train)
# input
sample_size, fea_num = X_train.shape
label_num = Y_train.shape[1]

X = tf.placeholder(tf.float32, [None, fea_num])
Y = tf.placeholder(tf.float32, [None, label_num])

W = tf.Variable(tf.zeros([fea_num, label_num]))
b = tf.Variable(tf.zeros([label_num]))

# logistic regression

logit_func = tf.matmul(X, W) + b
pred = tf.nn.sigmoid(logit_func)

# cross entropy cost
#cost_func = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred), 1))
# cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
cost_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_func,
                                                                   labels=Y))

#cost_func = tf.reduce_mean(tf.reduce_sum((-Y*tf.log(pred))-((1-Y)*tf.log(1-pred)), 1))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost_func)


init = tf.global_variables_initializer()

# run
#with tf.Session() as sess:
sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(sample_size/batch_size)

    for i in range(total_batch):
        begin_idx = batch_size*i
        batch_x = X_train.loc[begin_idx:begin_idx+batch_size, ]
        batch_y = Y_train.loc[begin_idx:begin_idx+batch_size, ]
        _, c = sess.run([optimizer, cost_func],
                        feed_dict={
                            X: batch_x.as_matrix(),
                            Y: batch_y.as_matrix()
                        })

        avg_cost += c / total_batch

    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "W={}".format(str(sess.run(W))),
              "b={}".format(str(sess.run(b))))

print("Optimization Finished!")

pred_result = sess.run(pred,
                       feed_dict={
                           X: X_test.as_matrix()
                       })
#print(X_test.as_matrix()*sess.run(W) + sess.run(b))

