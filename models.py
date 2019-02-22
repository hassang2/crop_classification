import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import tensorflow as tf
import utils

class KNN():
    def __init__(self, num_neighbors = 3):
        self.model = KNeighborsClassifier(n_neighbors = num_neighbors)
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, file_name):
        utils.pickle_it(self.model, file_name)
        print("Model saved")

    def load(self, file_name):
        self.model = utils.unpickle_it(file_name)
        print("Model loaded")
        
class KM():
    def __init__(self, num_clusters = 3):
        self.model = KMeans(n_clusters = num_clusters)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

class NN():
    def __init__(self, neurons = [5, 4, 3]):
        self.params = {}
        self.sess = tf.Session()
        self.neurons = neurons
        self.out = self.model(tf.placeholder(tf.float32, [None, neurons[0]], name = "X"))
        
    def model(self, X):
        L1 = tf.layers.dense(X, self.neurons[0])
        L2 = tf.layers.dense(L1, self.neurons[1])
        L3 = tf.layers.dense(L2, self.neurons[2])
        softmax = tf.nn.softmax(L3)
        return softmax

    def train(self, features, labels, validation_x = [], validation_y = [], epochs = 10, batch_size = 64):
        y_ = tf.placeholder(tf.float32, [None, 3])
        cost = tf.losses.softmax_cross_entropy(y_, self.out)
        train_step = tf.train.AdamOptimizer().minimize(cost)

        init = tf.initializers.global_variables()
        self.sess.run(init)

        for e in range(epochs):
            print("Epoch", e + 1)
            counter = 0
            for m in range(0, len(features), batch_size):
                self.sess.run(train_step, feed_dict = {"X:0": features[m:m+batch_size], y_: labels[m:m+batch_size]})
            if len(validation_x):
                validation_predict = self.predict(validation_x)
                correct = np.sum(validation_predict == validation_y)
                print("Epoch", e + 1, "accuracy:", correct / len(validation_x))
                self.save("model_saves/NN_model_e" + str(e+1) + ".ckpt")

    def save(self, file_name): 
        saver = tf.train.Saver()
        saver.save(self.sess, file_name)
        print("Model saved")

    def load(self, file_name):
        saver = tf.train.Saver()
        saver.restore(self.sess, file_name)
        print("Model loaded")

    def predict(self, features):
        return np.argmax(self.sess.run(self.out, feed_dict = {"X:0": features}), axis = 1)
