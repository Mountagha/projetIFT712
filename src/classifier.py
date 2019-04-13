# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

import pandas as pd
import sys
import argparse
import numpy as np
import methodsClassifier as mc
import dataManagement as gd
from datetime import datetime as time


def main():
    """
        A parser to allow user to easily test different methods with different parameters
    """
    parser = argparse.ArgumentParser(
    usage='\n python3 classifier.py [METHOD] [hyper_parameter_research] \n python3 classifier.py [METHOD] [method_parameters.....]',
    description='this program allows to test multiple classifiers methods on classification leaf data from kaggle')
    parser.add_argument('--method', type=str, default='SVC',
                    help='classifier to use. must be in [SVC, LDA, GaussianNB, MultinomialNB, KNN ,MLP]')
    parser.add_argument('--hyper_parameter_research', type=bool, default=False,
                    help='use hyper_parameter_research for model selection')
    parser.add_argument('--save', type=bool, default=False,
                    help='boolean to save file for submission or not')
    parser.add_argument('--kernel', type=str, default='rbf',
                    help='kernel for SVM classifier. must be in []')
    parser.add_argument('--C', type=float, default=float(10),
                    help='C parameter for SVM classifier')
    parser.add_argument('--degree', type=float, default=float(5),
                    help='degree for SVM classifier')
    parser.add_argument('--coef0', type=float, default=float(20),
                    help='coef0 for SVM classifier')
    parser.add_argument('--gamma', type=float, default=float(5),
                    help='gamma for SVM classifier')
    # LDA parameters
    parser.add_argument('--solver', type=str, default='svd',
                    help='solver for LDA classifier')
    # Gaussian Naive Bayes parameters
    parser.add_argument('--var_smoothing', type=float, default=float(1e-9),
                    help='var_smoothing for Gaussian Naive Bayes classifier')
    # Multinomial Naive Bayes parameters
    parser.add_argument('--alpha', type=float, default=float(0),
                    help='alpha for Multinomial Naive Bayes classifier')
    # Random forest parameters
    parser.add_argument('--n_estimators', type=float, default=int(100),
                    help='number of n_estimators for Multinomial Naive Bayes classifier')
    parser.add_argument('--max_features', type=str, default='auto',
                    help='max_features for  Random forest classifier ')
    # KNN parameters
    parser.add_argument('--n_neighbors', type=int, default=int(5),
                    help='number of n_neighbors for KNN classifier')
    parser.add_argument('--weights', type=str, default='uniform',
                    help='weights for KNN Classifier')
    parser.add_argument('--leaf_size', type=int, default=int(30),
                    help='size of leafs for KNN classifier')
    # MLP parameters
    parser.add_argument('--hidden_layer_sizes', type=tuple, default=(50,50),
                    help='number and size of layers as a tuple')
    parser.add_argument('--activation', type=str, default='tanh',
                    help='activation function of the network')
    parser.add_argument('--L2', type=float, default=float(1e-3),
                    help='L2 penalty for MLP classifier')
    parser.add_argument('--learning_rate_init', type=float, default=float(0.01),
                    help='value to initialize the learning_rate')

    # Method (SVC, LDA, GaussianNB, MultinomialNB)
    #method = 'SVC'


    args = parser.parse_args()

    method = args.method

    save = args.save

    # params for SVC

    kernel = args.kernel
    C = args.C
    degree = args.degree
    coef0 = args.coef0
    gamma = args.gamma
    # Params for LDA
    solver = args.solver
    # Params for Gaussian Naive Bayes (GaussianNB)
    var_smoothing = args.var_smoothing
    # Params for Multinomial Naive Bayes (MultinomialNB)
    alpha = args.alpha
    # Params for Random Forest (RF)
    n_estimators = args.n_estimators
    max_features = args.max_features
    # Params for KNN
    n_neighbors = args.n_neighbors
    weights = args.weights
    leaf_size = args.leaf_size
    # Params for MLP
    hidden_layer_sizes = args.hidden_layer_sizes
    activation = args.activation
    L2 = args.L2
    learning_rate_init = args.learning_rate_init

    # Data generator parameters
    normalization = True
    n_splits = int(10)
    data_size = float(0.2) # we train on 80% and test on 20% of the data
    #data_size = float(0.4)

    # parameters for hyper_parameter_research best in our case choose by experience
    r_hp = args.hyper_parameter_research
    n_iter_rs = 3
    cv_rs = 5
    cs_gs = 2

    # Create data generator and generate training and testing data
    dataManagement = gd.DataGenerator(n_splits, data_size, normalization)
    [x_train, t_train, x_test, t_test, tests, test_ids, classes] = dataManagement.generate_data()

    if method == 'SVC':
        classification = mc.Classification(r_hp, method, kernel, C, degree, coef0, gamma)
    elif method == 'LDA':
        classification = mc.Classification(r_hp, method, solver)
    elif method == 'GaussianNB':
        classification = mc.Classification(r_hp, method, var_smoothing)
    elif method == 'MultinomialNB':
        classification = mc.Classification(r_hp, method, alpha)
    elif method == 'RF':
        classification = mc.Classification(r_hp, method, n_estimators, max_features)
    elif method == 'KNN':
        classification = mc.Classification(r_hp, method, n_neighbors, weights, leaf_size)
    elif method == 'MLP':
        classification = mc.Classification(r_hp, method, hidden_layer_sizes, activation, L2, learning_rate_init)
    #classification = mc.Classification(r_hp, method, var_smoothing)

    # mesuring time execution
    tStart = time.now()
    classification.training(x_train, t_train)
    tEnd = time.now()
    c = tEnd - tStart

    # Predictions sur les ensembles d'entrainement et de test
    predictions_train = classification.prediction(x_train)
    predictions_test = classification.prediction(x_test)
    if save:
        predictions = classification.prediction(x_test, save=True)
        submission = pd.DataFrame(predictions, columns=classes)
        submission.insert(0, 'id', test_ids)
        submission.to_csv('../data/submission.csv', index=False)


    # erros and calculs
    training_acc = classification.accuracy(t_train, predictions_train) * 100
    err_idx_train = t_train[np.nonzero(t_train-predictions_train)]
    err_unique_idx_train = np.unique(err_idx_train)
    testing_acc = classification.accuracy(t_test, predictions_test) * 100
    err_idx_test = t_test[np.nonzero(t_test-predictions_test)]
    err_unique_idx_test = np.unique(err_idx_test)

    print('%%%%% Classifier : ', method, ' %%%%%')
    print('%%%%% Execution time :'+ str(c.seconds) + ' seconds' )
    if method == 'SVC':
        kernel = classification.get_params('kernel')
        print(' - kernel = ', kernel)
        print(' - C = ', '%.4f' % classification.get_params('C'))
        if kernel == 'poly':
            print(' - degree = ', '%.4f' % classification.get_params('degree'))
            print(' - coef0 = ', '%.4f' % classification.get_params('coef0'))
        elif kernel == 'rbf':
            print(' - gamma = ', '%.4f' % classification.get_params('gamma'))
        elif kernel == 'sigmoid':
            print(' - coef0 = ', '%.4f' % classification.get_params('coef0'))
    elif method == 'LDA':
        print(' - solver = ', classification.get_params('solver'))
    elif method == 'GaussianNB':
        print(' - var_smoothing = ,', classification.get_params('v_smoothing'))
    elif method == 'MultinomialNB':
        print(' - alpha = ', classification.get_params('alpha'))
    elif method == 'RF':
        print(' - n_estimators = ', classification.get_params('n_estimators'))
        print(' - max_features = ', classification.get_params('max_features'))
    elif method == 'KNN':
        print(' - n_neighbors = ', classification.get_params('n_neighbors'))
        print(' - weights = ', classification.get_params('weights'))
        print(' - leaf_size = ', classification.get_params('leaf_size'))
    elif method == 'MLP':
        print(' - hidden_layer_sizes = ', classification.get_params('hidden_layer_sizes'))
        print(' - activation = ', classification.get_params('activation'))
        print(' - alpha = ', classification.get_params('alpha'))


    print('')
    print('Training accuracy :', '%.2f' % training_acc, '%')
    print('# of training error : ', '%.2f' % len(err_idx_train))
    print('# of classes error : ', '%.2f' % len(err_unique_idx_train), '\n')
    print('Testing accuracy :', '%.2f' % testing_acc, '%')
    print('# of testing error : ', '%.2f' % len(err_idx_test))
    print('# of classes error : ', '%.2f' % len(err_unique_idx_test))


if __name__ == '__main__':
    main()
