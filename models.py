import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import tensorflow as tf

class KNN():
    def __init__(self, num_neighbors = 3):
        self.model = KNeighborsClassifier(n_neighbors = num_neighbors)
    
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class KM():
    def __init__(self, num_clusters = 3):
        self.model = KMeans(n_clusters = num_clusters)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

class NN():
    def __init__(self, neurons = [5, 64, 64, 3], num_classes = 3):
        self.params = {}
        # self.out = self.model([64, 64], 3)
        self.sess = tf.Session()
        self.out = None
        self.neurons = neurons
    def model(self, X, num_classes = 3):
        # depth = len(neurons)
        # for d in range(depth - 1):
        #     self.params['W' + str(d+1)] = tf.Variable(tf.truncated_normal([neurons[d], neurons[d+1]], stddev=0.01))
        #     self.layer_outs['y' + str(d+1)] = tf.nn.relu(tf.matmul(self.input, ))
        # X = tf.placeholder(tf.float32, [None, 1])
        
        L1 = tf.layers.dense(X, self.neurons[0])
        # before = L1
        # for n in self.neurons:
        #     before = tf.layers.dense(before, n)
        L2 = tf.layers.dense(L1, self.neurons[1])
        L3 = tf.layers.dense(L2, self.neurons[2])
        # L4 = tf.layers.dense(L3, self.neurons[3])
        softmax = tf.nn.softmax(L3)
        return softmax
    def train(self, features, labels, validation_x = [], validation_y = [], epochs = 10, batch_size = 64):
        y_ = tf.placeholder(tf.float32, [None, 3])
        X  = tf.placeholder(tf.float32, [None, 5], name = "X")
        y = self.model(X, 3)
        self.out = y
        cost = tf.losses.softmax_cross_entropy(y_, y)
        train_step = tf.train.AdamOptimizer().minimize(cost)

        init = tf.initializers.global_variables()
        self.sess.run(init)

        for e in range(epochs):
            print("Epoch", e+1)
            counter = 0
            for m in range(0, len(features), batch_size):
                self.sess.run(train_step, feed_dict = {X: features[m:m+batch_size], y_: labels[m:m+batch_size]})
            if len(validation_x):
                validation_predict = self.predict(validation_x)
                correct = np.sum(validation_predict == validation_y)
                print("Accuracy:", correct / len(validation_x))

        

    def predict(self, features):
        # X  = tf.placeholder(tf.float32, [None, 5])
        # y = self.model(X, [64, 64], 3)
        # init = tf.initializers.global_variables()
        # self.sess.run(init)

        predictions = np.zeros(len(features))
        for i in range(len(features)):
            predictions[i] = np.argmax(self.sess.run(self.out, feed_dict = {"X:0": [features[i]]}))
            # predictions.append(self.sess.run(self.out, feed_dict = {}))

        return predictions
        # return predictions
