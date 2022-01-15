# This code is written by Goutamkumar Tulajappa Kalburgi. NAU Email ID: gk325@nau.edu
__author__ = "Goutamkumar Tulajappa Kalburgi (gk325@nau.edu)"

import classifier

if __name__ == '__main__':
    """Digits Recognition"""
    # Creating an object of Classifier
    digitsClassifier = classifier.Classifier()
    # Loading the data from the file
    digitsClassifier.read_dataframe('optdigits.tra', 'last')
    # Splitting the data into train and test sets (Using half of the data at random to train the classifier)
    digitsClassifier.split_df_train_test(0.5)
    digitsClassifier.print_classes_occurrence()

    # Initializing the Gaussian Naive Bayes Classifier
    digitsClassifier.init_classifier('GaussianNaiveBayesClassifier')
    # Training the data
    digitsClassifier.train()
    # Predicting the features testing data
    digitsClassifier.predict_X_test()
    # Computing the accuracy of the predicted values
    digitsClassifier.score()
    digitsClassifier.get_priors()
    # Plot the data
    digitsClassifier.plot_and_predict_20_values()

    # Plotting the confusion matrix
    digitsClassifier.plot_confusion_matrix()

    """Letter Recognition"""
    lettersClassifier = classifier.Classifier()
    lettersClassifier.read_dataframe('letter-recognition.data', '0')
    lettersClassifier.split_df_train_test(0.5)
    lettersClassifier.print_classes_occurrence()

    lettersClassifier.init_classifier('GaussianNaiveBayesClassifier')
    lettersClassifier.train()
    lettersClassifier.predict_X_test()
    lettersClassifier.score()
    lettersClassifier.get_priors()
    lettersClassifier.plot_and_predict_20_values()

    lettersClassifier.plot_confusion_matrix()




