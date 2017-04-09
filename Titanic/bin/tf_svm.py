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
from ensemble import base_train, base_test, X_train, X_test, Y_train, id_test
import pdb
sys.path.append('../lib')
#from data import X_train, Y_train, X_test, id_test

Y_train = Y_train.replace(0, -1) # for svm training
enc = OneHotEncoder()
X_train = enc.fit_transform(X_train)
X_test = enc.fit_transform(X_test)


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

# # feature selection
# # train = whole_df[:train_rows]
# clf = ExtraTreesClassifier(n_estimators=200)
# clf = clf.fit(X_train, Y_train.Survived)
# # find out feature importance
# features = pd.DataFrame()
# # after one-hot encoded not care the columns name
# #features['feature'] = X_train.columns
# features['importance'] = clf.feature_importances_
# features.sort_values(by=['importance'],ascending=False)

# # only using top n features
# model = SelectFromModel(clf, prefit=True)
# X_train = model.transform(X_train)
# X_test =  model.transform(X_test)


# parameter setting
SVMC = 1
EPOCH_NUM = 2000
BATCH_SIZE = 200
LEARNING_RATE = 0.05
DISPLAY_STEP = 2


sample_size, feature_num = X_train.shape
label_num = Y_train.shape[1]

X = tf.placeholder(tf.float32, [None, feature_num])
Y = tf.placeholder(tf.float32, [None, label_num])
prediction_grid = tf.placeholder(tf.float32, [None, feature_num])
W = tf.Variable(tf.zeros([feature_num, label_num]))
#b = tf.Variable(tf.zeros([label_num]))
# Create variables for svm
b = tf.Variable(tf.random_normal(shape=[1, BATCH_SIZE]))


def cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs):
    """Tiles two tensors in perpendicular dimensions."""
    expanded_a = tf.expand_dims(tensor_a, 1)
    expanded_b = tf.expand_dims(tensor_b, 0)
    tiled_a = tf.tile(expanded_a, tf.constant([1, b_inputs, 1]))
    tiled_b = tf.tile(expanded_b, tf.constant([a_inputs, 1, 1]))

    return [tiled_a, tiled_b]



def linear_kernel(tensor_a, a_inputs, tensor_b, b_inputs):
    """Returns the linear kernel (dot product) matrix of two matrices of vectors
            element-wise."""
    cross = cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs)

    kernel = tf.reduce_sum(cross[0]*cross[1], reduction_indices=2)

    return kernel



def gaussian_kernel(tensor_a, a_inputs, tensor_b, b_inputs, gamma):
    """Returns the Gaussian kernel matrix of two matrices of vectors element-wise."""
    cross = cross_matrices(tensor_a, a_inputs, tensor_b, b_inputs)
    kernel = tf.exp(-tf.reduce_sum(tf.square(cross[0] - cross[1]), reduction_indices=2)*tf.constant(gamma, dtype=tf.float32))

    return kernel


def cost(training, classes, inputs, kernel_type="gaussian", C=1, gamma=1):
    """Returns the kernelised cost to be minimised."""
    beta = tf.Variable(tf.zeros([inputs, 1]))
    offset = tf.Variable(tf.zeros([1]))

    if kernel_type == "linear":
        kernel = linear_kernel(training, inputs, training, inputs)
    elif kernel_type == "gaussian":
        kernel = gaussian_kernel(training, inputs, training, inputs, gamma)

    x = tf.reshape(tf.div(tf.matmul(tf.matmul(
        beta, kernel, transpose_a=True), beta), tf.constant([2.0])), [1])

    y = tf.ones([1]) - classes*tf.add(tf.matmul(kernel, beta, transpose_a=True), offset)

    z = tf.reduce_mean(
        tf.reduce_max(
            tf.concat([tf.zeros_like(y), y], 1), reduction_indices=1))*tf.constant([C], dtype=tf.float32)

    cost = tf.add(x, z)

    return beta, offset, cost



def decide(training, training_instances, testing, testing_instances,
           beta, offset, kernel_type="gaussian", gamma=1):
    """Tests a set of test instances."""

    if kernel_type == "linear":
        kernel = linear_kernel(
            testing, testing_instances, training, training_instances)

    elif kernel_type == "gaussian":
        kernel = gaussian_kernel(
            testing, testing_instances, training, training_instances, gamma)

    return tf.sign(tf.add(tf.matmul(kernel, beta), offset))



beta, offset, cost = cost(X, Y, BATCH_SIZE, kernel_type="gaussian", C=1, gamma=1)


optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# linear SVM
#model_output = tf.matmul(X, W) + b

# RBF kernel
# lam = 1./2.
# alpha = tf.Variable(tf.random_uniform([BATCH_SIZE,1],-1.0,1.0))
# alpha = tf.maximum(0.,alpha)
# KX = tf.placeholder("float", shape=[BATCH_SIZE, BATCH_SIZE])
# y = tf.placeholder("float", shape=[BATCH_SIZE, 1])
# loss = lam*tf.reduce_sum(tf.matmul(alpha,tf.transpose(alpha))*KX)
# hinge = tf.reduce_mean(tf.maximum(0., 1. - y*tf.matmul(KX, alpha)))
# loss += hinge
# optimizer = tf.train.GradientDescentOptimizer(0.0002)
# train_op = optimizer.minimize(loss)



# # Gaussian (RBF) kernel
# gamma = tf.constant(-50.0)
# dist = tf.reduce_sum(tf.square(X), 1)
# dist = tf.reshape(dist, [-1,1])
# sq_dists = tf.add(
#     tf.subtract(dist,
#                 tf.multiply(2., tf.matmul(X, tf.transpose(X)))), tf.transpose(dist))
# my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))


# # Compute SVM Model
# first_term = tf.reduce_sum(b)
# b_vec_cross = tf.matmul(tf.transpose(b), b)
# y_target_cross = tf.matmul(Y, tf.transpose(Y))
# second_term = tf.reduce_sum(
#     tf.multiply(my_kernel,
#                 tf.multiply(b_vec_cross, y_target_cross)))
# loss = tf.negative(tf.subtract(first_term, second_term))

# optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)


# # Gaussian (RBF) prediction kernel
# rA = tf.reshape(tf.reduce_sum(tf.square(X), 1), [-1,1])
# rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1,1])
# pred_sq_dist = tf.add(
#     tf.subtract(rA, tf.multiply(2., tf.matmul(X, tf.transpose(prediction_grid)))), tf.transpose(rB))
# pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))

# prediction_output = tf.matmul(tf.multiply(tf.transpose(Y),b), pred_kernel)
# prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
# accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction), tf.squeeze(Y)), tf.float32))


# l2_norm = tf.reduce_sum(tf.square(W))
# regularization_loss = 0.001*tf.reduce_sum(tf.square(W))
# #hinge_loss = tf.reduce_mean(tf.maximum(tf.zeros([BATCH_SIZE, 1]), 1 - Y*model_output))
# hinge_loss = tf.reduce_mean(tf.maximum(0., 1 - Y*model_output))
# loss = regularization_loss + SVMC*hinge_loss

# pred = tf.sign(model_output)
# correct_pred = tf.equal(model_output, pred)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

#loss_vec = []
#with tf.Session() as sess:
sess = tf.Session()
sess.run(init)

for epoch in range(EPOCH_NUM):

    avg_cost = 0.
    total_batch = int(sample_size/BATCH_SIZE) # for the rest points
    #total_batch = int(sample_size/batch_size)

    for i in range(total_batch):
        begin_idx = BATCH_SIZE*i
        #print('begin idx:', begin_idx, '| end idx:', begin_idx+batch_size)
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
              # "W={}".format(sess.run(W)))
              # "cv={}".formate(cv_num))
              # "b={}".format(str(sess.run(b))))

    # rand_index = np.random.choice(X_train.shape[0], size=BATCH_SIZE)
    # rand_x = X_train[rand_index].toarray()
    # rand_y = Y_train.ix[rand_index].as_matrix()
    # sess.run(optimizer, feed_dict={X: rand_x, Y: rand_y})

    # temp_loss = sess.run(loss, feed_dict={X: rand_x, Y: rand_y})
    # # loss_vec.append(temp_loss)

    # if (epoch+1)%100==0:
    #     print('Step #' + str(epoch + 1))
    #     print('Loss = ' + str(temp_loss))

print("Optimization Finished!")

test = X_test.toarray()
test_tensor = tf.placeholder(tf.float32, [None, feature_num])

# Classifies a test point from the trained SVM parameters.
model = decide(
    X, X_train.shape[0], test_tensor, X_test.shape[0], beta, offset, kernel_type="gaussian",
    gamma=1)

print("Test data classified as signal: %f%%" % sess.run(
    tf.reduce_sum(model), feed_dict={X: X_train.toarray(),
                                     test_tensor: test}))


pred_result = sess.run(prediction,
                       feed_dict={
                           X: X_.toarray(),
                           prediction_grid: X_test.toarray()
                       })

result_df =  pd.concat([id_test, pd.DataFrame(pred_result)], axis=1)
result_df.columns = ["PassengerId" , "Survived"]
result_df.loc[(result_df['Survived'] >= 0.5), 'Survived'] = 1
result_df.loc[(result_df['Survived'] < 0.5), 'Survived'] = 0
result_df['Survived'] = result_df.Survived.apply(int)
result_df.to_csv('../data/svm_result_to_submission', index=False)
