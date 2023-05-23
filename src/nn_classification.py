'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 2: Image classification benchmarks

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
The script contains code to classify the cifar10 dataset using a neural network.

'''

# import packages
import os
import cv2
import argparse
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

# define arguments that can be set by the user from command line
def argument_parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("--hidden_layer_sizes", type=int, nargs='+', help='Specify size of hidden layers')
        ap.add_argument("--learning_rate", help="How to update the weights. Can be 'constant', 'invscaling' or 'adaptive'", default = 'adaptive')
        ap.add_argument("--early_stopping", type=bool, help= "Stop if validation score is not improving", default = 'False')
        ap.add_argument("--max_iter", type=int, help="Maximum number of iterations", default = 15)
        ap.add_argument("--clf_report_name", help="what the name of the classification report should be", default='nn_clf_report.txt')
        args = vars(ap.parse_args())

        return args

def train_nn_clf(X_train, y_train, X_test, hidden_layer_sizes, learning_rate, early_stopping, max_iter):
        '''
        Trains a neural network classifier using training data and predicts from the model using test data.
        Returns predicted y values

        Arguments:
        - X_train: np-array with train data
        - y_train: np-array with train data labels
        - X_test: np-array with test data
        - hidden_layer_sizes: sizes of hidden layers in the neural network
        - learning_rate: Learning rate schedule for weight updates
        - early_stopping: if true, training stops if validation score is not improving
        - max_iter: Maximum number of iterations

        Returns: 
        List of predicted classes
        
        '''
        
        # train neural network
        clf = MLPClassifier(random_state=2830,
                    hidden_layer_sizes=tuple(hidden_layer_sizes),
                    learning_rate=learning_rate,
                    early_stopping=early_stopping,
                    verbose=True,
                    max_iter=max_iter).fit(X_train, y_train)

        # predict data
        y_pred = clf.predict(X_test)

        return y_pred

def create_report(y_pred, y_true, labels, filename):
        '''
        Create classification report from predicted and true class labels.
        Saves report in the 'out' folder

        Arguments:
        - y_pred: predicted classes for test data
        - y_true: actual class labels for test data
        - labels: label names of the classes
        - filename: what to call the classification report

        Returns:
        None
        '''
        # create classification report
        report = classification_report(y_true,
                                        y_pred,
                                        target_names = labels)

        # define output path
        path = os.path.join("out", filename)

        # save report
        with open(path, 'w') as file:
                file.write(report)

def main():
        
        args = argument_parser()

        # from the data.py script, import data loading function
        from data import data_loader

        (X_train, y_train), (X_test, y_test), labels = data_loader()

        # train model, predict from test data and save predictions
        y_pred = train_nn_clf(X_train, y_train, X_test, args['hidden_layer_sizes'], args['learning_rate'], args['early_stopping'], args['max_iter'])

        # save classification report
        create_report(y_pred, y_test, labels, args['clf_report_name'])

if __name__ == '__main__':
   main()