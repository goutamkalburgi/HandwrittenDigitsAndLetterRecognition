# This code is written by Goutamkumar Tulajappa Kalburgi. NAU Email ID: gk325@nau.edu
__author__ = "Goutamkumar Tulajappa Kalburgi (gk325@nau.edu)"

import numpy as np


class GaussianNaiveBayesClassifier:
    """
    A class that implements Gaussian Naive Bayes Classifier

    ...

    Attributes
    ----------
    classes: numpy.array
        target values
    mean: numpy.array
        mean for each target class's feature
    var: numpy.array
        variance for each target class's feature
    priors: numpy.array
        P(y), A target's probability
    accuracy: float
        mean accuracy on the supplied test data and labels.

    Methods
    -------
    fit(X, y, epsilon=1e-5)
        Trains the model to the training data input
    predict(X)
        Makes predictions on the features data input
    compute_predict(X)
        Determines the posterior probability of each class
    probability_density_function(class_index, features_data)
        Calculates the Gaussian Distribution for the data input
    score(y_true, y_prediction)
        Return the mean accuracy based on the test data and labels provided
    get_priors()
        The prior probability array is returned

    """

    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
        self.accuracy = None
        self.class_priors = {}

    def fit(self, X, y, epsilon=1e-5):
        """The model is trained using the training data input

        Parameters
        ----------
        X: numpy.array
            Features training data
        y: numpy.array
            Target training data
        epsilon: float, optional
            If the variance for a target class feature is 0, you can use the epsilon value instead (default is 1e-5)
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        # compute each class's mean, variance, and prior
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0) + epsilon
            self.priors[idx] = X_c.shape[0] / float(n_samples)
            self.class_priors[c] = self.priors[idx]

    def predict(self, X):
        """Predicts based on the data inputted features

        Parameters
        ----------
        X: numpy.array
            Features data

        Returns
        -------
        numpy.array
            An array of predictions on the features data input
        """
        y_prediction = [self.compute_predict(x) for x in X]
        return np.array(y_prediction)

    def compute_predict(self, x):
        """Determines the posterior probability of each class

        Parameters
        ----------
        x: numpy.array
            Features data

        Returns
        -------
        class
            Class with the highest posterior probability
        """
        posteriors = []

        # compute each class's posterior probability
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.probability_density_function(idx, x), where=(self.probability_density_function(idx, x) != 0)))
            posterior = prior + posterior
            posteriors.append(posterior)

        # choose the class with the highest posterior probability and return it
        return self.classes[np.argmax(posteriors)]

    def probability_density_function(self, class_idx, x):
        """Calculates the Gaussian Distribution for the data input

        Parameters
        ----------
        class_idx: int
            class index
        x: numpy.array
            Features data

        Returns
        -------
        float
            Gaussian Distribution function for the data input
        """
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def score(self, y_true, y_prediction):
        """Return the mean accuracy based on the test data and labels provided

        Parameters
        ----------
        y_true: numpy.array
            Target data
        y_prediction: numpy.array
            Predicted target data

        Returns
        -------
        float
            The mean accuracy based on the test data and labels provided
        """
        self.accuracy = np.sum(y_true == y_prediction) / len(y_true)
        return self.accuracy

    def get_priors(self):
        """The prior probability array is returned"""
        return self.class_priors
