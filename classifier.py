'''
Submission by: Hassan Goodarzifarahani

-Cut 140 from top, 200 from left, 20 from bottom, and 100 from right
-the dimensions of the dataset image AFTER croppings: (5799, 9125, 5)
Possible BUGS:
- 
'''

import numpy as np
import utils
import models
import time
from random import shuffle
import matplotlib.pyplot as plt
import imageio
from skimage.external import tifffile

TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.05
FROM_PICKLE = False

if FROM_PICKLE:
    train_data   = utils.unpickle_it('pickles/train_data')
    train_labels = utils.unpickle_it('pickles/train_labels')
    test_labels  = utils.unpickle_it('pickles/test_labels')
    test_data    = utils.unpickle_it('pickles/test_data')
    validation_labels = utils.unpickle_it('pickles/validation_labels')
    validation_data   = utils.unpickle_it('pickles/validation_data')
    south_predictions = utils.unpickle_it('pickles/predictions')

    print("All data loaded from pickles")

else:
    train_data, test_data, validation_data, train_labels, test_labels, validation_labels = utils.process_data(TEST_FRACTION, VALIDATION_FRACTION, True)
    print("data processed")

    utils.pickle_it(train_data, 'pickles/train_data')
    utils.pickle_it(test_data, 'pickles/test_data')
    utils.pickle_it(validation_data, 'pickles/validation_data')
    utils.pickle_it(train_labels, 'pickles/train_labels')
    utils.pickle_it(test_labels, 'pickles/test_labels')
    utils.pickle_it(validation_labels, 'pickles/validation_labels')
    print("data pickled")


######################################################
# K-Means
# model = models.KM(254)
# model.fit(train_data)

######################################################
# K-Nearest Neighbors
# model = models.KNN(3)
# model.fit(train_data, train_labels)
# model.save("model_saves/KNN")
# model.load("model_saves/KNN")

######################################################
# Deep Learning
train_labels = utils.make_one_hot(train_labels, 3)
model = models.NN(neurons = [5, 4, 3])
model.train(train_data, train_labels, validation_data, validation_labels, epochs = 8)
# model.save("model_saves/NN_model.ckpt")
# model.load("model_saves/NN_model_e7.ckpt")

######################################################
# Test set accuracy for the model
test_size = len(test_data)
test_predictions = model.predict(test_data[0 : test_size])
correct = np.sum(test_predictions == test_labels[0 : test_size])
print("Accuracy:", correct / test_size)

######################################################
# Labeling the south part of the image
south_img = utils.extract_data("data/test_south.tif")
south_img_shape = np.shape(south_img)
south_data = south_img.reshape(shape = (np.shape(south_img)[0] * np.shape(south_img)[1], np.shape(south_img)[2]))
south_data = utils.standardize(south_data)
south_predictions = model.predict(south_data)
south_predictions_img = predictions.reshape((south_img_shape[0], south_img_shape[1]))

## applying a denoising filter:
# south_predictions_img = utils.denoise(south_predictions_img)
south_predictions_img = utils.to_RGB(south_predictions_img)

plt.imshow(south_predictions_img)
plt.axis('off')
plt.show()

