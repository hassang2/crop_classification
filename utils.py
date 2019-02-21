import numpy as np
import imageio
import pickle
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_bilateral
from random import shuffle




def extract_data(file_name):
    return np.asarray(imageio.mimread(file_name, memtest = False))[0]

def split_data(data, fractions):
    indices = [int(np.sum(fractions[:i+1]) * len(data)) for i in range(len(fractions))]
    return np.array_split(data, indices)

'''
Splits and returns train and test features and labels.
@test_fraction fraction of the data used for testing. (rest is for training)
@return 4 numpy arrays in this order: train_data, test_data, train_labels, test_labels
'''
def process_data(test_fraction = 0.15, validation_fraction = 0.0,  limit_classes = True):
    data_path = "data/train_north.tif"
    truth_path = "data/truth_north.tif"
    prediction_path  = "data/test_south.tif"

    img  = extract_data(data_path)
    img  = cut_borders(img, [140, 200, 20, 100])
    data = np.reshape(img, (np.shape(img)[0] * np.shape(img)[1], np.shape(img)[2]))
    

    labels = extract_data(truth_path)
    labels = cut_borders(labels, [140, 200, 20, 100])
    labels = labels.flatten()
    labels = prune_labels(labels, [1, 5])

    data, labels = shuffle_data(data, labels)
    [test_data, validation_data, train_data] = split_data(data, [test_fraction, validation_fraction])
    [test_labels, validation_labels, train_labels] = split_data(labels, [test_fraction, validation_fraction])

    train_data = standardize(train_data)
    test_data  = standardize(test_data)
    validation_data = standardize(validation_data)

    return train_data, test_data, validation_data, train_labels, test_labels, validation_labels


def pickle_it(data, file_name):
    pickling_on = open(file_name, 'wb')
    pickle.dump(data, pickling_on)
    pickling_on.close()

def unpickle_it(file_name):
    try:
        pickle_off = open(file_name, 'rb')
        stuff = pickle.load(pickle_off)
        return stuff
    except:
        return None

def make_one_hot(labels, n_labels):
    z = np.zeros((len(labels), n_labels))
    z[np.arange(len(labels)), labels] = 1
    return z

def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

def arr_to_img(arr):
    # colors = [
    # # 'yellow':
    # [254, 211, 0],
    # # 'green' :
    # [38, 212, 0],
    # # 'brown' : 
    # [89, 67, 29],
    # ]
    colors = [
    # 'yellow':
    [1.0, 0.82745098039, 0],
    # 'green' :
    [0.14901960784, 0.43921568628, 0],
    # 'brown' : 
    [0.35294117647, 0.25490196078, 0.11372549019],
    ]

    img = np.zeros((np.shape(arr)[0], np.shape(arr)[1], 3))

    for i in range(len(arr)):
        for j in range(len(arr[0])):
            img[i][j] = colors[arr[i][j]]
    return img

def shuffle_data(features, labels):
    indices = np.arange(len(features))
    np.random.shuffle(indices)

    return features[indices], labels[indices]

'''
@dims how much to cut from each side. It is read in this order: [top, left, bottom, right] 
@return image vector after cut
'''
def cut_borders(data, dims):
    return data[dims[0] : len(data) - dims[2] , dims[1]: len(data[0]) - dims[3]]

def prune_labels(labels, tags = [1, 5]):
    np.place(labels, labels == 1, [0])
    np.place(labels, labels == 5, [1])
    np.place(labels, labels > 1, [2])
    return labels



