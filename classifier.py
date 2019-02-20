'''
-Cut 140 from top, 200 from left, 20 from bottom, and 100 from right
-There are 254 classes
-the dimensions of the dataset image AFTER croppings: (5799, 9125, 5)
Possible BUGS:
- 

TODO:
-break long line
-shuffle data
-put code in functions in classes
-custom data directories for extract_data function
-validation set for NN
-have to NNs . one to recognize corn and the other soy
-remove prediction function loop
-standardize validation data

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
TEST_FRACTION = 0.3
VALIDATION_FRACTION = 0.0001
from_pickle = True
if from_pickle:
    train_data   = utils.unpickle_it('pickles/small_train_data')
    test_data    = utils.unpickle_it('pickles/small_test_data')
    validation_data = utils.unpickle_it('pickles/small_validation_data')
    train_labels = utils.unpickle_it('pickles/small_train_labels')
    test_labels  = utils.unpickle_it('pickles/small_test_labels')
    validation_labels = utils.unpickle_it('pickles/small_validation_labels')
    print("All data loaded from pickles")

else:
    train_data, test_data, validation_data, train_labels, test_labels, validation_labels = utils.extract_data(TEST_FRACTION, VALIDATION_FRACTION, True)
    print("data created")
    train_data = utils.standardize(train_data)
    test_data  = utils.standardize(test_data)
    # zipped = list(zip(train_data, train_labels))
    # shuffle(zipped)
    # train_data, train_labels = zip(*zipped)
    # train_data = list(train_data)
    # train_labels = list(train_labels)
    # print("zipped")
    utils.pickle_it(train_data[0:num], 'pickles/small_train_data')
    utils.pickle_it(test_data[0:test_size], 'pickles/small_test_data')
    utils.pickle_it(validation_data[0:num], 'pickles/small_validation_data')
    utils.pickle_it(train_labels[0:num], 'pickles/small_train_labels')
    utils.pickle_it(test_labels[0:test_size], 'pickles/small_test_labels')
    utils.pickle_it(validation_labels[0:num], 'pickles/small_validation_labels')
    print("data pickled")
    # exit()


######################################################
# model = models.KM(254)
# model.fit(train_data)
######################################################
# model = models.KNN(3)
# model.fit(train_data, train_labels)
######################################################
train_labels = utils.make_one_hot(train_labels, 3)
model = models.NN(neurons = [5, 4, 3])
# model.train(train_data, train_labels, validation_data, validation_labels, epochs = 5)
model.train(train_data, train_labels, epochs = 4)

######################################################
predictions = model.predict(test_data[0 : test_size])
# truncated = predictions[:(test_size // 9125) * 9125]
# truncated = truncated.reshape(-1, 9125)
# # denoised  = denoise_bilateral(truncated)
# denoised = denoised.flatten()
correct = np.sum(predictions == test_labels[0 : test_size])
# correct_denoised = np.sum(denoised == test_labels[0 : test_size])


print(correct, " out of ", test_size)
# print(correct_denoised, " denoised ", test_size)
print(correct / test_size)
exit()
img = []
print(np.shape(predictions))
for i in range(test_size // 9125):
    print(i)
    truncated = predictions[i*9125 : (i+1)*9125]
    # truncated = truncated.reshape(-1, 9125)
    img.append(truncated)
    # if i == 8:
    #     print(np.shape(img))
    #     plt.axis('off')
    #     plt.imshow(img)
    #     plt.show()
    #     exit()

plt.axis('off')
plt.imshow(img)
plt.show()



