from re import L
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_zoom
import matplotlib.pyplot as plt

def preprocess(array, shape):
    """
    Normalizes the supplied array and reshpaed it into the appropriate format.
    """ 
    array= np.array(array)
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), shape[0],shape[1],shape[2]))
    return array

def noise(array):
    """
    Adds random noise to each images in the supplied array.
    """

    noise_factor = 0.2
    noisy_array = array + noise_factor * np.random.normal(size=array.shape)

    return np.clip(noisy_array, 0.0, 1.0)

def augment(array, labels=None, ratio=0.05):
    """
    Adds random augment to each images in the supplied array as ratio%
    """
    aug_X = []
    aug_Y = []
    for i in range(int(ratio*len(array))):
        data_aug_idx = random.randrange(0, len(array)-1)
        img=random_rotation(array[data_aug_idx], rg=80, row_axis=0, col_axis=1, channel_axis=2)
        aug_X.append(img)
        if labels is not None:
            aug_Y.append(labels[data_aug_idx])

        data_aug_idx = random.randrange(0, len(array)-1)
        img=random_shift(array[data_aug_idx], wrg=0.1, hrg=0.1, row_axis=0, col_axis=1, channel_axis=2)
        aug_X.append(img)
        if labels is not None:
            aug_Y.append(labels[data_aug_idx])

        data_aug_idx = random.randrange(0, len(array)-1)
        img=random_zoom(array[data_aug_idx], zoom_range=[0.6,0.9], row_axis=0, col_axis=1, channel_axis=2)
        aug_X.append(img)
        if labels is not None:
            aug_Y.append(labels[data_aug_idx])

    aug_X = np.array(aug_X)
    aug_Y = np.array(aug_Y)
    print("Augmentation samples are" ,len(aug_X))

    if labels is not None:
        return np.concatenate((array, aug_X), axis=0), np.concatenate((labels, aug_Y), axis=0)
    else:
        return np.concatenate((array, aug_X), axis=0)

def tsne_plot(x1, labels_as_one_column, fig_title=None):
    """
    tsne analysis for viewing
    """

    x1 = x1.reshape(x1.shape[0], -1)
    print("Before Shape is: ", x1.shape)

    pca_50 = PCA(n_components = 50)
    pca_results_50 = pca_50.fit_transform(x1)

    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(pca_results_50)

    print("After Shape is: ", X_t.shape)

    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[:, 0], X_t[:, 1], s= 5, c=labels_as_one_column, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar()
    
    plt.legend(loc='best')
    plt.title(fig_title)
    plt.show()

def display_pair(array1, array2, label=None):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size = n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    if label is not None:
        print("Label: " , np.array(label[indices]))

    plt.figure(figsize=(20,4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n , i+1)
        plt.imshow(image1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2,n, i+1+n)
        plt.imshow(image2)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()
    print(array1.shape, array2.shape)

def shuffler(X,Y=None):
    
    if Y is not None:
        # Shuffled Image
        temp_pool = list(zip(X,Y))
        random.shuffle(temp_pool)
        X, Y = zip(*temp_pool)
        return np.array(X), np.array(Y)

    else:
        random.shuffle(X)
        return X

def label_dict_static(classifier):
    if classifier == "OOP":
        ## center(c0), behind(c7), phoning(c1), close(c5), far(c6)
        label_map={'c6': 0, 'c5': 1, 'c7': 2, 'c1': 3, 'c0': 4}
        label_str={'c6': "Far", 'c5': "Close", 'c7': "Behind", 'c1': "Phone", 'c0': "Center"}
    
    elif classifier == "Belt":
        #벨트(b0: 0), 노벨트(b1: 1)
        label_map = {'b0': 0, 'b1': 1}
        label_str={'b0': "Belt", 'b1':"Unbelt"}

    elif classifier == "Mask":
        #(m0: 마스크, 0), (m1: 노마스크, 1)
        label_map = {'m0': 0, 'm1': 1}
        label_str={'m0': "Mask", 'm1':"Nomask"}

    elif classifier == "Weak":
        #남자(s0: 0), 여자(s1: 1)
        label_map = {'s0': 0, 's1': 1}
        label_str={'s0': "Man", 's1':"Woman"}

    return label_map, label_str 

def score(preds, labels):
    ret=0
    for pred, label in zip(preds, labels):

        pred_idx = np.argmax(pred)
        label_idx = np.argmax(label)

        if (pred_idx == label_idx):
            ret+=1

    ret = ret / len(preds)
    print("Test Predict is: {}%".format(ret*100))

    return ret