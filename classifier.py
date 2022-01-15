# This code is written by Goutamkumar Tulajappa Kalburgi. NAU Email ID: gk325@nau.edu
__author__ = "Goutamkumar Tulajappa Kalburgi (gk325@nau.edu)"

import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from gaussiannaivebayesclassifier import GaussianNaiveBayesClassifier
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class Classifier:
    """
    A class used to represent a Classifier

    ...

    Attributes
    ----------
    df : pandas.DataFrame
        dataframe
    X : numpy.array
        Features data
    y : numpy.array
        target data
    X_train : numpy.array
        Features training data
    X_test : numpy.array
        Features testing data
    y_train : numpy.array
        Target training data
    y_test : numpy.array
        Target testing data
    y_predicted : numpy.array
        Predicted target data
    classes : numpy.array
        target values
    model : GaussianNaiveBayesClassifier
        Type of Classifier


    Methods
    -------
    read_dataframe(file_name, target_features_index)
        Load a comma-separated values (csv) file into DataFrame and split features and target data
    split_df_train_test(test_size)
        Split array into train and test subsets at random based on the test_size
    print_classes_occurrence()
         Prints the count of each class in your training and testing target data
    init_classifier(classifier_type)
        Creates the classifier model object
    train()
        Calls the classifier's fit method
    predict_X_test()
        Calls the classifier's predict method for features testing data and prints the predicted target data
    score()
        Calls the classifier's score method and prints the mean accuracy based on the test data and labels provided
    plot_and_predict_20_values()
        Plots first 20 values from the features data and print the predicted target data for the shown feature plot
    plot_confusion_matrix()
        Plots the confusion matrix
    getPriors(range)
        Calls the classifier's get_priors method and prints the prior for each class based on range

    """

    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_predicted = None
        self.classes = None
        self.model = None

    def read_dataframe(self, file_name, target_index):
        """Load a comma-separated values (csv) file into DataFrame and split features and target data

        Parameters
        ----------
        file_name : str
            dataframe file name
        target_index :
            Index of the target column in the dataframe, it could be 0 (first column) or 'last'(last column)
        """
        self.df = pd.read_csv(file_name)
        if target_index == '0':
            self.X = self.df.iloc[:, 1:].to_numpy()
            self.y = self.df.iloc[:, 0].to_numpy()
        else:
            self.X = self.df.iloc[:, :-1].to_numpy()
            self.y = self.df.iloc[:, -1].to_numpy()

    def split_df_train_test(self, test_size):
        """Split array into train and test subsets at random based on the test_size

        Parameters
        ----------
        test_size : float
            Represents the percent of the dataset that should be included in the train split
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)

    def print_classes_occurrence(self):
        """Prints the count of each class in your training and testing target data"""
        y_train_number_count = {}
        y_test_number_count = {}
        self.classes = np.unique(self.y)
        for i in self.classes:
            y_train_number_count[i] = np.count_nonzero(self.y_train == i)
            y_test_number_count[i] = np.count_nonzero(self.y_test == i)
        print("The count of each class in training set is")
        for k, v in y_train_number_count.items():
            print("Count of " + str(k) + " is " + str(v))
        print("The count of each class in testing set is")
        for k, v in y_test_number_count.items():
            print("Count of " + str(k) + " is " + str(v))

    def init_classifier(self, classifier_type):
        """Creates the classifier model object

        Parameters
        ----------
        classifier_type : str
            Type of Classifier
        """
        if classifier_type == 'GaussianNaiveBayesClassifier':
            self.model = GaussianNaiveBayesClassifier()

    def train(self):
        """Calls the classifier's fit method"""
        self.model.fit(self.X_train, self.y_train)

    def predict_X_test(self):
        """Calls the classifier's predict method for features testing data and prints the predicted target data"""
        self.y_predicted = self.model.predict(self.X_test)
        print("The predicted value for X_test(testing data) is " + str(self.y_predicted))

    def score(self):
        """Calls the classifier's score method and prints the mean accuracy based on the test data and labels provided"""
        print("The mean accuracy based on the test data and labels provided is " + str(self.model.score(self.y_predicted, self.y_test)))

    def plot_and_predict_20_values(self):
        """Plots first 20 values from the features data and print the predicted target data for the shown feature plot"""
        for i in range(20):
            arr = self.X_test[i]
            no_of_features = self.X_test.shape[1]
            n = int(math.sqrt(no_of_features))
            arr_2d = np.reshape(arr, (n, n))
            plt.imshow(arr_2d, cmap='binary')
            plt.show()
            print("The true value is " + str(self.y_test[i]))
            print("The predicted value is " + str(self.model.predict([self.X_test[i]])))

    def plot_confusion_matrix(self):
        """Plots the confusion matrix"""
        cm = confusion_matrix(self.y_test, self.y_predicted)
        plt.figure(figsize=(26, 26))
        sn.heatmap(cm, annot=True, cmap="OrRd")
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()

    def get_priors(self):
        """Calls the classifier's get_priors method"""
        prob = self.model.get_priors()
        for k,v in prob.items():
            print("Probability of " + str(k) + " is " + str(v))












