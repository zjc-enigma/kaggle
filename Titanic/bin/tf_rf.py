import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators \
    import estimator

from tensorflow.contrib.learn.python.learn\
        import metric_spec
from tensorflow.contrib.tensor_forest.client\
        import eval_metrics
from tensorflow.contrib.tensor_forest.client\
        import random_forest
from tensorflow.contrib.tensor_forest.python\
        import tensor_forest

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.platform import app
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# mnist = input_data.read_data_sets('../data', one_hot=False)
# t_x = mnist.train.images
# t_y = mnist.train.labels

import pandas as pd

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

sample_size, fea_num = X_train.shape
#label_num = Y_train.shape[1]
label_num = 2

#sample_size = 3
#fea_num = 3
#label_num = 2
model_dir = '../data/'
# hyper parameter settings
n_trees = 100
use_training_loss = False
max_nodes = 1000
batch_size = 50

#FLAGS = None

def build_estimator(model_dir):
    """Build an estimator."""
    params = tensor_forest.ForestHParams(
        num_classes=label_num,
        num_features=fea_num,
        num_trees=n_trees,
        max_nodes=max_nodes)

    graph_builder_class = tensor_forest.RandomForestGraphs
    if use_training_loss:
        graph_builder_class = tensor_forest.TrainingLossForest

    # Use the SKCompat wrapper, which gives us a convenient way to split
    # in-memory data like MNIST into batches.
    return estimator.SKCompat(
        random_forest.TensorForestEstimator(
            params,
            graph_builder_class=graph_builder_class,
            model_dir=model_dir))


est = build_estimator(model_dir)
est.fit(x=X_train.todense(),
        y=Y_train.Survived.as_matrix(),
        batch_size=batch_size)



hparams = tensor_forest.ForestHParams(
    num_trees=n_trees,
    max_nodes=max_nodes,
    num_classes=2,
    num_features=fea_num
)


classifier = random_forest.TensorForestEstimator(hparams)
X = X_train.toarray().astype(np.float32)
Y = Y_train.Survived.as_matrix().astype(np.float32)
T = X_test.toarray().astype(np.float32)
classifier.fit(x=X, y=Y, steps=20)
#classifier.evaluate(x=X, y=Y, steps=10)

result_list = classifier.predict(x=T)
result_df = pd.concat([id_test, pd.DataFrame(result_list)], axis=1)
result_df.columns = ["PassengerId" , "Survived"]
result_df['Survived'] = result_df.Survived.apply(int)
result_df.to_csv('../data/rf_result_to_submission', index=False)
