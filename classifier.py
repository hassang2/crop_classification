'''
-Cut 140 from top, 200 from left, 20 from bottom, and 100 from right
-There are 254 classes
-the dimensions of the dataset image AFTER croppings: (5799, 9125, 5)
Possible BUGS:
- 

TODO:
-break long line
-put code in functions in classes
-custom data directories for extract_data function
-CLI
-create image

'''

import numpy as np
import utils
import models
import time
from random import shuffle
import matplotlib.pyplot as plt
from skimage.restoration import denoise_bilateral

num = 1000000
test_size = 10000
valid_size = 10000
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.05
from_pickle = True
if from_pickle:
    # train_data   = utils.unpickle_it('pickles/small_train_data')
    # test_data    = utils.unpickle_it('pickles/small_test_data')
    # validation_data = utils.unpickle_it('pickles/small_validation_data')
    # train_labels = utils.unpickle_it('pickles/small_train_labels')
    # test_labels  = utils.unpickle_it('pickles/small_test_labels')
    # validation_labels = utils.unpickle_it('pickles/small_validation_labels')

    train_data   = utils.unpickle_it('pickles/train_data')
    test_data    = utils.unpickle_it('pickles/test_data')
    validation_data = utils.unpickle_it('pickles/validation_data')
    train_labels = utils.unpickle_it('pickles/train_labels')
    test_labels  = utils.unpickle_it('pickles/test_labels')
    validation_labels = utils.unpickle_it('pickles/validation_labels')

    print("All data loaded from pickles")

else:
    train_data, test_data, validation_data, train_labels, test_labels, validation_labels = utils.process_data(TEST_FRACTION, VALIDATION_FRACTION, True)
    print("data processed")

    utils.pickle_it(train_data[0:num], 'pickles/small_train_data')
    utils.pickle_it(test_data[0:test_size], 'pickles/small_test_data')
    utils.pickle_it(validation_data[0:valid_size], 'pickles/small_validation_data')
    utils.pickle_it(train_labels[0:num], 'pickles/small_train_labels')
    utils.pickle_it(test_labels[0:test_size], 'pickles/small_test_labels')
    utils.pickle_it(validation_labels[0:valid_size], 'pickles/small_validation_labels')

    # utils.pickle_it(train_data, 'pickles/train_data')
    # utils.pickle_it(test_data, 'pickles/test_data')
    # utils.pickle_it(validation_data, 'pickles/validation_data')
    # utils.pickle_it(train_labels, 'pickles/train_labels')
    # utils.pickle_it(test_labels, 'pickles/test_labels')
    # utils.pickle_it(validation_labels, 'pickles/validation_labels')
    print("data pickled")
    exit()


######################################################
# model = models.KM(254)
# model.fit(train_data)
######################################################
# model = models.KNN(3)
# model.fit(train_data, train_labels)
# model.save("model_saves/KNN")
# model.load("model_saves/KNN")
######################################################
train_labels = utils.make_one_hot(train_labels, 3)
model = models.NN(neurons = [5, 4, 3])
# model.train(train_data, train_labels, validation_data, validation_labels, epochs = 8)
# model.train(train_data, train_labels, epochs = 8)
# model.save("model_saves/NN_model.ckpt")
model.load("model_saves/NN_model_e7.ckpt")
######################################################
test_size = len(test_data)

predictions = model.predict(test_data[0 : test_size])
correct = np.sum(predictions == test_labels[0 : test_size])

south_img = utils.extract_data("data/test_south.tif")
south_data = np.reshape(south_img, (np.shape(south_img)[0] * np.shape(south_img)[1], np.shape(south_img)[2]))
south_data = utils.standardize(south_data)
south_predicitons = model.predict(south_data)

utils.pickle_it(south_predicitons, "pickles/predictions")


