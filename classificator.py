# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

import sys
import numpy as np
import methods_classificator as mc
import data_generator as gd


def warning(erreur_test, erreur_apprentissage):
    """
    Fonction qui affiche un WARNING à l'ecran lorsque les erreurs obtenues en fonction du bruit
    indique une possibilite de sur- ou de sous-apprentissage
    """
    # AJOUTER CODE ICI


def main():
    # input: n_splits, data_size, normalization, r_hp, method, params...
    # if sys.argv[5] not in ['SVC', 'LDA', 'GaussianNB', 'MultinomialNB', 'KNN', 'MLP']:
    #     print('Classifier not implemented, choose between : [SVC, LDA, GaussianNB, MultinomialNB, KNN, MLP]')
    #     return
    # else:
    #     if sys.argv[5] in ['LDA', 'GaussianNB', 'MultinomialNB'] and len(sys.argv) != 6:
    #         print('Wrong numbre of input')
    #         return
    #     elif sys.argv[5] in ['RF'] and len(sys.argv) != 7:
    #         print('Wrong numbre of input')
    #         return
    #     elif sys.argv[5] in ['SVC'] and len(sys.argv) != 10:
    #         print('Wrong numbre of input')
    #         return

    # n_splits = int(sys.argv[1])
    # data_size = float(sys.argv[2])
    # normalization = bool(sys.argv[3])
    # r_hp = bool(sys.argv[4])
    # method = sys.argv[5]

    # Method (SVC, LDA, GaussianNB, MultinomialNB)
    method = 'RF'

    # Params for SVC
    kernel = 'rbf'
    C = float(10)
    degree = float(5)
    coef0 = float(20)
    gamma = float(5)
    # Params for LDA
    solver = 'svd'
    # Params for Gaussian Naive Bayes (GaussianNB)
    var_smoothing = float(1e-9)
    # Params for Multinomial Naive Bayes (MultinomialNB)
    alpha = float(0)
    # Params for Random Forest (RF)
    n_estimators = int(100)
    max_features = 'auto'

    # Data generator parameters
    normalization = False
    n_splits = int(10)
    data_size = float(0.2)

    # Hyper parameters research
    r_hp = bool(1)

    # Create data generator and generate training and testing data
    data_generator = gd.DataGenerator(n_splits, data_size, normalization)
    [x_train, t_train, x_test, t_test] = data_generator.generate_data()

    # Entrainement du modele de regression
    # classification = mc.Classification(method, kernel, C, degree, coef0, gamma, solver, var_smoothing, alpha,
    #                                    n_estimators, max_features)
    classification = mc.Classification(method, n_estimators, max_features)

    if r_hp:
        classification.hyperparametre_research(x_train, t_train)
    else:
        classification.training(x_train, t_train)

    # Predictions sur les ensembles d'entrainement et de test
    predictions_train = classification.prediction(x_train)
    predictions_test = classification.prediction(x_test)

    # Calcul des erreurs
    training_error = classification.error(t_train, predictions_train) * 100
    err_idx_train = t_train[np.nonzero(t_train-predictions_train)]
    err_unique_idx_train = np.unique(err_idx_train)
    testing_error = classification.error(t_test, predictions_test) * 100
    err_idx_test = t_test[np.nonzero(t_test-predictions_test)]
    err_unique_idx_test = np.unique(err_idx_test)

    # Normalize and replace 0 by 1e-15 (removes divergence problem if divided by prob) for submission file
    #for i in range(len(predictions_test)):
        #predictions_test = max(min(predictions_test[i], 1 - 1e-15), 1e-15)

    print('%%%%% Classifier : ', method, ' %%%%%')
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
    if method == 'LDA':
        print(' - solver = ', classification.get_params('solver'))
    if method == 'GaussianNB':
        print(' - var_smoothing = ,', classification.get_params('v_smoothing'))
    if method == 'MultinomialNB':
        print(' - alpha = ', classification.get_params('alpha'))
    if method == 'RF':
        print(' - n_estimators = ', classification.get_params('n_estimators'))
        print(' - max_features = ', classification.get_params('max_features'))


    print('')
    print('Training error :', '%.2f' % training_error, '%')
    print('# of training error : ', '%.2f' % len(err_idx_train))
    print('# of classes error : ', '%.2f' % len(err_unique_idx_train), '\n')
    print('Testing error :', '%.2f' % testing_error, '%')
    print('# of testing error : ', '%.2f' % len(err_idx_test))
    print('# of classes error : ', '%.2f' % len(err_unique_idx_test))

    warning(testing_error, training_error)


if __name__ == '__main__':
    main()
