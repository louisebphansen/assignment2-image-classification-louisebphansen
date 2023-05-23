'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 2: Image classification benchmarks

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
The script contains code to load and preprocess the cifar10 dataset

'''
# import packages
import cv2
import numpy as np
from tensorflow.keras.datasets import cifar10


def prep_data(X_train:np.ndarray, X_test:np.ndarray) -> np.ndarray:
    '''
    This function preprocesses input training and testing image data to prepare it for scikit-learns classification methods.
    The preprocessing steps consists of converting the images to greyscale, scaling them and reshaping them.

    Arguments:
    - X_train: a numpy array containing the training data
    - X_test: a numpy array containing the testing data

    Returns:
    - Preprocessed numpy arrays X_train and X_test datasets ready to train a classifier.

    '''

    # convert each of the images to greyscale
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

    # scale the images by dividing by 255
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0

    # reshape the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))

    return X_train_dataset, X_test_dataset
    
def data_loader():
        ''' 
        This functions loads the cifar10 dataset and preprocesses it.

        Arguments:
        -None

        Returns:
        Preprocessed X and y data (np.arrays) ready for a classifier, as well as class labels
        ''' 

        # load X and y data
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # manually define labels of the 10 classes
        labels = ["airplane", 
                "automobile", 
                "bird", 
                "cat", 
                "deer", 
                "dog", 
                "frog", 
                "horse", 
                "ship", 
                "truck"]

        # prep data using function
        X_train_data, X_test_data = prep_data(X_train, X_test)

        return (X_train_data, y_train), (X_test_data, y_test), labels
