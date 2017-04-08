import tensorflow as tf
import functools

def lazy_property(function):

    attribute = '_' + function.__name__
    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper

class NonlinearSVC(object):
    def __init__(self,
                 learning_rate=0.001,
                 training_epoch=None,
                 error=0.001,
                 display_step=5):
        self.learning_rate = learning_rate
        self.training_epoch = training_epoch
        self.display_step = display_step
        self.error = error

    def __Preprocessing(self, trainX):
        row = trainX.shape[0]
        col = trainX.shape[1]
        self.X = tf.placeholder(shape=[row, col], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[row, 1], dtype=tf.float32)
        self.test = tf.placeholder(shape=[None, col], dtype=tf.float32)
        self.beta = tf.Variable(tf.truncated_normal(shape=[1, row], stddev=.1))

    @lazy_property
    def Kernel_Train(self):
        tmp_abs = tf.reshape(tensor=tf.reduce_sum(tf.square(self.X), axis=1), shape=[-1,1])
        tmp_ = tf.add(tf.sub(tmp_abs, tf.mul(2., tf.matmul(self.X, tf.transpose(self.X)))), tf.transpose(tmp_abs))
        return tf.exp(tf.mul(self.gamma, tf.abs(tmp_)))

    @lazy_property
    def Kernel_Prediction(self):
        tmpA = tf.reshape(tf.reduce_sum(tf.square(self.X), 1),[-1,1])
        tmpB = tf.reshape(tf.reduce_sum(tf.square(self.test), 1),[-1,1])
        tmp = tf.add(tf.sub(tmpA, tf.mul(2.,tf.matmul(self.X, self.test, transpose_b=True))), tf.transpose(tmpB))
        return tf.exp(tf.mul(self.gamma, tf.abs(tmp)))

    @lazy_property
    def Cost(self):
        left = tf.reduce_sum(self.beta)
        beta_square = tf.matmul(self.beta, self.beta, transpose_a=True)
        Y_square = tf.matmul(self.Y, self.Y, transpose_b= True)
        right = tf.reduce_sum(tf.mul(self.Kernel_Train, tf.mul(beta_square, Y_square)))
        return tf.neg(tf.sub(left, right))

    @lazy_property
    def Prediction(self):
        kernel_out = tf.matmul(tf.mul(tf.transpose(self.Y),self.beta), self.Kernel_Prediction)
        return tf.sign(kernel_out - tf.reduce_mean(kernel_out))


    @lazy_property
    def Accuracy(self):
        return tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(self.Prediction), tf.squeeze(self.Y)), tf.float32))

    def fit(self, trainX, trainY, gamma=50.):
        self.sess = tf.InteractiveSession()
        self.__Preprocessing(trainX)
        self.gamma = tf.constant(value=-gamma, dtype=tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.Cost)
        self.sess.run(tf.global_variables_initializer())

        if self.training_epoch is not None:
            for ep in range(self.training_epoch):
                self.sess.run(self.optimizer, feed_dict={self.X:trainX, self.Y:trainY})
                if ep % self.display_step== 0:
                    loss, acc = self.sess.run([self.Cost, self.Accuracy], feed_dict={self.X:trainX, self.Y:trainY, self.test:trainX})
                    print ('epoch=',ep,'loss= ',loss, 'accuracy= ', acc)
        elif self.training_epoch is None:
            acc = 0.1
            ep = 0
            while (acc< 1.- self.error):
                acc,_ = self.sess.run([self.Accuracy, self.optimizer], feed_dict={self.X:trainX, self.Y:trainY, self.test:trainX})
                ep += 1
                if ep % self.display_step== 0: 
                    loss = self.sess.run(self.Cost, feed_dict={self.X:trainX, self.Y:trainY})
                    print ('epoch=',ep,'loss= ',loss, 'accuracy= ', acc)
        print("Optimization Finished!")
        self.trainX = trainX
        self.trainY = trainY
    
    def pred(self,test):
        output = self.sess.run(self.Prediction, feed_dict={self.X:self.trainX, self.Y:self.trainY, self.test:test})
        return output
