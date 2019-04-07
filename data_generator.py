# -*- coding: utf-8 -*-

#####
# Marc-Antoine Genest (14 079 588)
# Mamadou Mountagha Bah (18 098 915)
#####

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


class DataGenerator:
    def __init__(self, n_splits, data_size):
        self.n_splits = n_splits
        self.data_size = data_size

    def generate_data(self):
        """
        Function that returns training and testing data
        """

        # Read current path (containing training and testing data sets)
        dir_path = os.path.dirname(os.path.abspath(__file__))

        # Read training and testing data sets
        train = pd.read_csv(os.path.join(dir_path, 'train.csv'))
        test = pd.read_csv(os.path.join(dir_path, 'test.csv'))

        # Organize data sets
        train, labels, test, test_ids, classes = self.encode(train, test)

        # Split data sets (data sets are too large (990 * data_size))
        sss = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.data_size, random_state=20)
        for train_index, test_index in sss.split(train, labels):
            x_train, x_test = train.values[train_index], train.values[test_index]
            t_train, t_test = labels[train_index], labels[test_index]

        return x_train, t_train, x_test, t_test

    def encode(self, train, test):
        """
        Swiss army knife function to organize data sets
        """
        # Initialize encoder from sklearn
        le = LabelEncoder().fit(train.species)

        # Encode species strings
        labels = le.transform(train.species)

        # Save column names for submission
        classes = list(le.classes_)

        # Save test ids for submission
        test_ids = test.id

        train = train.drop(['species', 'id'], axis=1)
        test = test.drop(['id'], axis=1)

        return train, labels, test, test_ids, classes