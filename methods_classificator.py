# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

import numpy as np
import random
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


class Classification:
    def __init__(self, method, kernel, c, prob):
        # Method to use
        self.method = method

        # Initialize classifier
        self.classifier = None

        # SVC method
        self.kernel = kernel
        self.C = c
        self.probability = prob

    def hyperparametre_research(self, X, t):
        """

        """

    def training(self, x_train, t_train):
        """

        """
        # Choose classifier
        if self.method == 'SVC':
            self.classifier = SVC(kernel=self.kernel, C=self.C, probability=self.probability)

        elif self.method == 'LDA':
            self.classifier = LinearDiscriminantAnalysis()

        # Train classifier
        self.classifier.fit(x_train, t_train)

    def prediction(self, x_test):
        """

        """
        # Predict leaf from x_test
        prediction = self.classifier.predict(x_test)

        return prediction

    @staticmethod
    def error(t, prediction):
        """

        """
        # Calculate error with sklearn metric function
        err = (1 - accuracy_score(t, prediction, normalize=True))*100

        return err
