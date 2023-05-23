'''
VISUAL ANALYTICS @ AARHUS UNIVERSITY, ASSIGNMENT 2: Image classification benchmarks

AUTHOR: Louise Brix Pilegaard Hansen

DESCRIPTION:
The script contains code to classify the cifar10 dataset using a multinomial logistic regression.

'''

# import packages
import os
import cv2
import argparse
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# define arguments that can be set by the user from command line
def argument_parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("--penalty", help="Regularization penalty. Can be None, 'l2', 'l1' or 'elasticnet'.", default='None')
        ap.add_argument("--tol", type=float, help="Tolerance for stopping criteria", default=0.1)
        ap.add_argument("--solver", help="The algorithm to be optimized. See different options and limitations on sklearn documentation", default='saga')
        ap.add_argument("--clf_report_name", help="what the name of the classification report should be", default='log_clf_report.txt')
        args = vars(ap.parse_args())

        return args

def train_log_clf(X_train, y_train, X_test, penalty, tol, solver):
        '''
        Trains a multinomial logistic classifier using train data and predicts from the model using the test data

        Arguments:
        - X_train: np-array with train data
        - y_train: np-array with train data labels
        - X_test: np-array with test data
        - penalty: regularization penalty
        - tol: tolerance for stopping criteria
        - solver: optimization algorithm (only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss, see sklearn documentation)

        Returns:
        List of predicted classes
        '''
        
        # train multinomial classifier and fit on training data
        clf = LogisticRegression(penalty=penalty,
                                tol=tol, 
                                verbose=True, 
                                solver=solver, 
                                multi_class="multinomial").fit(X_train, y_train)
        # predict data using model
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

        # train model, predict from test data and save preidictions
        y_pred = train_log_clf(X_train, y_train, X_test, args['penalty'], args['tol'], args['solver'])

        # save classification report
        create_report(y_pred, y_test, labels, args['clf_report_name'])
       
       
if __name__ == '__main__':
   main()