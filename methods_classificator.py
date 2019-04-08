# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

# Math. tools
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

# Models
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# sklearn tools
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score


class Classification:
    def __init__(self, method, kernel, c, degree, coef0, gamma, solver, var_smoothing, alpha, n_estimators,
                 max_features):
        # Method to use
        self.method = method

        # Initialize classifier
        self.classifier = None

        # SVC method
        self.kernel = kernel
        self.C = c
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

        # LDA method
        self.solver = solver

        # Gaussian Naive Bayes (GaussianNB) method
        self.v_smoothing = var_smoothing

        # Multinomial Naive Bayes (MultinomialNB) method
        self.alpha = alpha

        # Random Forest (RF) method
        self.n_estimators = n_estimators
        self.max_features = max_features

        # Choose classifier
        if self.method == 'SVC':
            self.classifier = SVC(kernel=self.kernel, C=self.C, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        elif self.method == 'LDA':
            self.classifier = LinearDiscriminantAnalysis(solver=self.solver)
        elif self.method == 'GaussianNB':
            self.classifier = GaussianNB()
        elif self.method == 'MultinomialNB':
            self.classifier = MultinomialNB(alpha=self.alpha)
        elif self.method == 'RF':
            self.classifier = RandomForestClassifier(n_estimators=self.n_estimators, max_features=self.max_features)

    def rand_distribution_hr(self):
        param_dist = {}
        if self.method == 'SVC':
            param_dist.update({'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                               'C': sp_randint(1, 50),
                               'degree': sp_randint(1, 10),
                               'coef0': sp_randint(1, 50),
                               'gamma': sp_randint(1, 20)})
        elif self.method == 'MultinomialNB':
            param_dist.update({'alpha': sp_uniform(0, 5)})  # (min_value, max_value - min_value)
        elif self.method == 'RF':
            param_dist.update({'n_estimators': sp_randint(10, 100),
                               'max_features': ['log2', 'sqrt', 1.0]})

        return param_dist

    def grid_distribution_hr(self, rs, sweep_test, n_test):
        param_dist = {}
        if self.method == 'SVC':
            best_rand_kernel = rs.best_estimator_.get_params()['kernel']
            best_rand_c = rs.best_estimator_.get_params()['C']
            c_dist = np.linspace(np.max([1, best_rand_c - sweep_test]), best_rand_c + sweep_test, n_test)
            param_dist.update({'kernel': [best_rand_kernel], 'C': c_dist})
            if best_rand_kernel == 'poly':
                best_rand_d = rs.best_estimator_.get_params()['degree']
                best_rand_c0 = rs.best_estimator_.get_params()['coef0']
                d_dist = np.linspace(np.max([1, best_rand_d - sweep_test]), best_rand_d + sweep_test, n_test)
                c0_dist = np.linspace(best_rand_c0 - sweep_test, best_rand_c0 + sweep_test, n_test)
                param_dist.update({'degree': d_dist, 'coef0': c0_dist})
            elif best_rand_kernel == 'rbf':
                best_rand_gamma = rs.best_estimator_.get_params()['gamma']
                gamma_dist = np.linspace(np.max([0, best_rand_gamma - sweep_test]), best_rand_gamma + sweep_test,
                                         n_test)
                param_dist.update({'gamma': gamma_dist})
            elif best_rand_kernel == 'sigmoid':
                best_rand_c0 = rs.best_estimator_.get_params()['coef0']
                c0_dist = np.linspace(best_rand_c0 - sweep_test, best_rand_c0 + sweep_test, n_test)
                param_dist.update({'coef0': c0_dist})
        elif self.method == 'LDA':
            param_dist.update({'solver': ['svd', 'lsqr', 'eigen']})
        elif self.method == 'MultinomialNB':
            best_rand_alpha = rs.best_estimator_.get_params()['alpha']
            param_dist.update({'alpha': np.linspace(np.max([0, best_rand_alpha - 1]), best_rand_alpha + 1, n_test)})
        elif self.method == 'RF':
            best_rand_n = rs.best_estimator_.get_params()['n_estimators']
            param_dist.update({
                'n_estimators': np.linspace(np.max([10, best_rand_n - sweep_test]), best_rand_n + sweep_test, n_test, dtype=int),
                'max_features': [rs.best_estimator_.get_params()['max_features']]})

        return param_dist

    def hyperparametre_research(self, X, t):
        """

        """
        # Find a first approximation of hyperparameters with a randomsearch
        param_dist = self.rand_distribution_hr()
        random_search = RandomizedSearchCV(self.classifier, refit=True, param_distributions=param_dist,
                                           n_iter=3, cv=3)
        if self.method not in ['LDA', 'GaussianNB']:
            random_search.fit(X, t)

        # Fine tune research with a gridsearch
        param_dist = self.grid_distribution_hr(random_search, 5, 5)
        grid_search = GridSearchCV(self.classifier, param_grid=param_dist, cv=2)
        grid_search.fit(X, t)

        # Set new classifier as the gridsearch, since he is fitted and can predict data
        self.set_params(grid_search)

    def get_params(self, param):
        return getattr(self, param)

    def set_params(self, clf):
        self.classifier = clf
        if self.method == 'SVC':
            self.kernel = self.classifier.best_estimator_.kernel
            self.C = self.classifier.best_estimator_.C
            self.degree = self.classifier.best_estimator_.degree
            self.coef0 = self.classifier.best_estimator_.coef0
            self.gamma = self.classifier.best_estimator_.gamma
        elif self.method == 'LDA':
            self.solver = self.classifier.best_estimator_.solver
        elif self.method == 'MultinomialNB':
            self.alpha = self.classifier.best_estimator_.alpha

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
