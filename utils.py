import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from random import shuffle
from skimage.restoration import denoise_bilateral
from skimage.external import tifffile

'''
processes the data. Operations include: extract data from given files, cut unnecessary borders,
prune labels, shuffle data, standardize data, and cut them into given portions.
@test_fraction fraction of the data used for testing.
@validation_fraction fraction of the data used for validation.
@return 6 numpy arrays in this order: train_data, test_data, validation_data, train_labels, test_labels, validation_labels
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

'''
pickles data
'''
def pickle_it(data, file_name):
    pickling_on = open(file_name, 'wb')
    pickle.dump(data, pickling_on)
    pickling_on.close()

'''
unpickles data
'''
def unpickle_it(file_name):
    try:
        pickle_off = open(file_name, 'rb')
        stuff = pickle.load(pickle_off)
        return stuff
    except:
        return None

'''
turns the list of labels into a lit of one hot vectors
'''
def make_one_hot(labels, n_labels):
    z = np.zeros((len(labels), n_labels))
    z[np.arange(len(labels)), labels] = 1
    return z

'''
Standardizes the data
'''
def standardize(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)

'''
Turns labels into RGB values
'''
def to_RGB(arr):
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

'''
Shuffles the data
'''
def shuffle_data(features, labels):
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    return features[indices], labels[indices]

'''
cuts borders from the image
@dims how much to cut from each side. It is read in this order: [top, left, bottom, right] 
@return image vector after cut
'''
def cut_borders(img, dims):
    return img[dims[0] : len(img) - dims[2] , dims[1]: len(img[0]) - dims[3]]

'''
limits labels to the given ones 
@labels the list of labels we want limit
@tags the labels we want to keep. everything else will be labeled with the same label.
@return pruned list of labels
'''
def prune_labels(labels, tags = [1, 5]):
    np.place(labels, labels == 1, [0])
    np.place(labels, labels == 5, [1])
    np.place(labels, labels > 1, [2])
    return labels

'''
Applies a denoising algorithm to the given image.
@img the image we want to denoise
@return the denoised image
'''
def denoise(img):
    for i in range(1, len(img) - 1):
        for j in range(1, len(img[i]) - 1):
            unique, counts = np.unique(img[i-1:i+2, j-1:j+2], return_counts=True)
            for u in range(len(unique)):
                if counts[u] >= 6:
                    img[i][j] = unique[u]
    return img

'''
extract data from the given file address
'''
def extract_data(file_name):
    return tifffile.imread(file_name)

'''
splits the given data into multiple portions stated in the @fractions argument.
'''
def split_data(data, fractions):
    indices = [int(np.sum(fractions[:i+1]) * len(data)) for i in range(len(fractions))]
    return np.array_split(data, indices)
