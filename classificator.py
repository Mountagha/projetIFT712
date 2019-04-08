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

#    if len(sys.argv) < 8:
#        print('Usage: python regression.py sk modele_gen nb_train nb_test bruit M lambda\n')
#        print('\t sk=0: using_sklearn=False, sk=1: using_sklearn=True')
#        print('\t modele_gen=lineaire, sin ou tanh')
#        print('\t nb_train: nombre de donnees d'entrainement')
#        print('\t nb_test: nombre de donnees de test')
#        print('\t bruit: amplitude du bruit appliqué aux données')
#        print('\t M: degré du polynome de la fonction de base (recherche d'hyperparametre lorsque M<0) ')
#        print('\t lambda: lambda utilisé par le modele de Ridge\n')
#        print(' exemple: python3 regression.py 1 sin 20 20 0.3 10 0.001\n')
#        return

#    method = sys.argv[1]
#    r_hp = bool(sys.argv[2])
#    n_splits = int(sys.argv[3])
#    data_size = float(sys.argv[4])
#    m = int(sys.argv[6])
#    lamb = float(sys.argv[7])
    # Method (SVC, LDA, GaussianNB, MultinomialNB)
    method = 'RF'

    # Params for SVC
    kernel = 'rbf'
    c = float(10)
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
    data_size = float(0.5)

    # Hyper parameters research
    r_hp = bool(1)

    # Create data generator and generate training and testing data
    data_generator = gd.DataGenerator(n_splits, data_size, normalization)
    [x_train, t_train, x_test, t_test] = data_generator.generate_data()

    # Entrainement du modele de regression
    classification = mc.Classification(method, kernel, c, degree, coef0, gamma, solver, var_smoothing, alpha,
                                       n_estimators, max_features)
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
