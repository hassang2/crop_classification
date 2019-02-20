import numpy as np
import imageio
import pickle
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_bilateral
'''
Splits and returns train and test features and labels.
@test_fraction fraction of the data used for testing. (rest is for training)
@return 4 numpy arrays in this order: train_data, test_data, train_labels, test_labels
'''
def extract_data(test_fraction = 0.15, validation_fraction = 0.0,  limit_classes = True):
    train_path = "data/train_north.tif"
    truth_path = "data/truth_north.tif"
    test_path  = "data/test_south.tif"
    table_path = "data/color_table.dbf"

    train_img  = np.asarray(imageio.mimread(train_path, memtest = False))[0]
    train_img  = train_img[140 : len(train_img) - 20 , 200: len(train_img[0]) - 100]

    train_img_shape = np.shape(train_img)
    data       = np.reshape(train_img, (train_img_shape[0] * train_img_shape[1], train_img_shape[2]))

    data_split = np.array_split(data, [int(len(data) * test_fraction), int(len(data) * (test_fraction + validation_fraction))])

    test_data       = data_split[0]
    validation_data = data_split[1]
    train_data      = data_split[2]
    # print(train_img_shape)
    # print(np.shape(train_data))
    # print(np.shape(test_data))
    # exit()
    labels = np.asarray(imageio.mimread(truth_path, memtest=False))[0]
    labels = labels[140 : len(labels) - 20 , 200: len(labels[0]) - 100]
    labels = labels.flatten()

    if limit_classes:
        np.place(labels, labels == 5, [0])
        np.place(labels, labels > 1, [2])

    labels_split = np.split(labels, [int(len(labels) * test_fraction), int(len(data) * (test_fraction + validation_fraction))])

    test_labels       = labels_split[0]
    validation_labels = labels_split[1]
    train_labels      = labels_split[2]
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

