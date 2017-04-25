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


# enc = OneHotEncoder()
# X_train = enc.fit_transform(X_train)
# X_test = enc.fit_transform(X_test)

sample_size, fea_num = X_train.shape
label_num = Y_train.shape[1]


# parameters
learning_rate = 0.05
training_epochs = 1000
batch_size = 200
display_step = 10
kfold_split_num = 8
gpu_num = 2
regularizer_beta = 0.005
layer_num = 1

# Network Parameters
n_hidden_1 = 10 # 1st layer number of features
n_hidden_2 = 10 # 2nd layer number of features
n_input = fea_num
n_classes = label_num


X = tf.placeholder(tf.float32, [None, fea_num])
Y = tf.placeholder(tf.float32, [None, label_num])


# Store layers weight & bias
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

}
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))

}


# W = tf.Variable(tf.zeros([fea_num, label_num]))
# b = tf.Variable(tf.zeros([label_num]))



# Create model
def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    #layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Output layer with sigmoid activation
    out_layer = tf.nn.sigmoid(tf.matmul(layer_2, weights['out']) + biases['out'])
    return out_layer


# Construct model
pred = multilayer_perceptron(X, weights, biases)
#l2_regularizer = regularizer_beta*tf.nn.l2_loss(W)


all_w = tf.concat([tf.reshape(mat, [-1]) for mat in weights.values()], axis=0)
regularizer = regularizer_beta*tf.reduce_mean(tf.square(all_w))

# Define loss and optimizer
cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=Y)) + regularizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables

init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(sample_size/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            begin_idx = batch_size*i
            batch_x = X_train[begin_idx:begin_idx+batch_size].toarray()
            batch_y = Y_train[begin_idx:begin_idx+batch_size].as_matrix()

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x,
                                                          Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    
    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    pred_result = sess.run(pred,
                           feed_dict={
                               X: X_test.toarray()
                           })


    result_df = pd.concat([id_test, pd.DataFrame(pred_result)], axis=1)
    result_df.columns = ["PassengerId" , "Survived"]
    result_df.loc[(result_df['Survived'] >= 0.5), 'Survived'] = 1
    result_df.loc[(result_df['Survived'] < 0.5), 'Survived'] = 0
    result_df['Survived'] = result_df.Survived.apply(int)
    result_df.to_csv('../data/nn_result_to_submission', index=False)
