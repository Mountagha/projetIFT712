# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.model_selection import StratifiedShuffleSplit


class DataGenerator:
    def __init__(self, n_splits, data_size, norm):
        self.train = None
        self.test = None
        self.n_splits = n_splits
        self.data_size = data_size
        self.norm = norm

    def generate_data(self):
        """
        Function that returns training and testing data
        """

        # Read current path (containing training and testing data sets)
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Read training and testing data sets
        self.train = pd.read_csv(os.path.join(dir_path, '../data/train.csv'))
        self.test = pd.read_csv(os.path.join(dir_path, '../data/test.csv'))

        # save original data for eventual later use
        train = self.train
        test  = self.test

        train, labels, tests, test_ids, classes = self.encode(train, test)

        # Split data sets (data sets are too large (990 * data_size))
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.data_size, random_state=20)
        for train_index, test_index in sss.split(train, labels):
            x_train, x_test = train.values[train_index], train.values[test_index]
            t_train, t_test = labels[train_index], labels[test_index]

        # Normalize data if asked by user
        if self.norm:
            Normalizer().fit_transform(x_train)
            Normalizer().fit_transform(x_test)

        return x_train, t_train, x_test, t_test, tests, test_ids, classes

    def encode(self, train, test):
        """
        Swiss army knife function to organize data sets
        """
        # Initialize encoder from sklearn
        train = self.train
        test  = self.test

        le = LabelEncoder().fit(train.species)

        # Encode species strings
        labels = le.transform(train.species)

        # Save column names for submission
        classes = list(le.classes_.shape)

        # Save test ids for submission
        test_ids = test.id

        train = train.drop(['species', 'id'], axis=1)
        test = test.drop(['id'], axis=1)

        return train, labels, test, test_ids, classes
