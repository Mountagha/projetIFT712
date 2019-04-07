# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

# Math. tools
import numpy as np
from scipy.stats import randint as sp_randint

# Models
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# sklearn tools
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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

        # Choose classifier
        if self.method == 'SVC':
            self.classifier = SVC(kernel=self.kernel, C=self.C, probability=self.probability)
        elif self.method == 'LDA':
            self.classifier = LinearDiscriminantAnalysis()


    def hyperparametre_research(self, X, t):
        """

        """
        # Find a first approximation of hyperparameters with a randomsearch
        if self.method == 'SVC':
            param_dist = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'C': sp_randint(1, 50),
                          'degree': sp_randint(1, 10),
                          'coef0': sp_randint(1, 50),
                          'gamma': sp_randint(1, 20)}

        # Search for hyperparameters
        if self.method != 'LDA':
            n_iter_search = 3
            cv = 3
            random_search = RandomizedSearchCV(self.classifier, refit=True, param_distributions=param_dist, n_iter=n_iter_search, cv=cv)
            random_search.fit(X, t)

        # Fine tune research with a gridsearch
        sweep_test = 5
        n_test = 5
        if self.method == 'SVC':
            best_rand_kernel = random_search.best_estimator_.get_params()['kernel']
            best_rand_c = random_search.best_estimator_.get_params()['C']
            c_dist = np.linspace(np.max([1, best_rand_c - sweep_test]), best_rand_c + sweep_test, n_test)
            param_dist = {'kernel': [best_rand_kernel], 'C': c_dist}
            if best_rand_kernel == 'poly':
                best_rand_d = random_search.best_estimator_.get_params()['degree']
                best_rand_c0 = random_search.best_estimator_.get_params()['coef0']
                d_dist = np.linspace(np.max([1, best_rand_d - sweep_test]), best_rand_d + sweep_test, n_test)
                c0_dist = np.linspace(best_rand_c0 - sweep_test, best_rand_c0 + sweep_test, n_test)
                param_dist.update({'degree': d_dist, 'coef0': c0_dist})
            elif best_rand_kernel == 'rbf':
                best_rand_gamma = random_search.best_estimator_.get_params()['gamma']
                gamma_dist = np.linspace(np.max([0, best_rand_gamma - sweep_test]), best_rand_gamma + sweep_test, n_test)
                param_dist.update({'gamma': gamma_dist})
            elif best_rand_kernel == 'sigmoid':
                best_rand_c0 = random_search.best_estimator_.get_params()['coef0']
                c0_dist = np.linspace(best_rand_c0 - sweep_test, best_rand_c0 + sweep_test, n_test)
                param_dist.update({'coef0': c0_dist})
        elif self.method == 'LDA':
            param_dist = {'solver': ['svd', 'lsqr', 'eigen']}

        # Search for hyperparameters
        cv = 2
        grid_search = GridSearchCV(self.classifier, param_grid=param_dist, cv=cv)
        grid_search.fit(X, t)

        # Set new classifier as the gridsearch, since he is fitted and can predict data
        self.classifier = grid_search

    def getParams(self, param):
        if param == 'C':
            return self.classifier.best_estimator_.C
        elif param == 'gamma':
            return self.classifier.best_estimator_.gamma
        elif param == 'kernel':
            return self.classifier.best_estimator_.kernel
        elif param == 'coef0':
            return self.classifier.best_estimator_.coef0
        elif param == 'degree':
            return self.classifier.best_estimator_.degree
        elif param == 'solver':
            return self.classifier.best_estimator_.solver

    def training(self, x_train, t_train):
        """

        """
        # Train classifier
        self.classifier.fit(x_train, t_train)

    def prediction(self, x_test):
        """

        """
        # Returns prediction a data set
        prediction = self.classifier.predict(x_test)

        return prediction

    @staticmethod
    def error(t, prediction):
        """

        """
        # Calculate accuracy with sklearn metric function
        acc_score = accuracy_score(t, prediction, normalize=True)

        return (1 - acc_score)
