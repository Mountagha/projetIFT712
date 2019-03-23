
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

CHEMIN_DONNEES = '/home/mountagha/projetIFT712'

class gestion_donnees:
    """
    Cette classe permet de gérer tout ce qui est en rapport avec le traitement des données
    du chargement en passant par le pretraitement jusqu'à l'obtention des données prêtes à être
    chargées dans le model d'entrainement
    """
    def __init__(self):
        self.X_entrainement = None
        self.Y_entrainement = None
        self.X_test = None
        self.classes = None
        self.idTests = None


    """
    Cette fonction permet de charger les donnees à partir d'un fichier CSV
    params : nom: le nom du fichier csv à charger
    """

    def chargement_donnees(self, nom=None):
        if nom is None:
            print("Vous devez spécifiez quelles donnees vous voulez charger")
            return
        chemin = os.path.join(CHEMIN_DONNEES, nom)
        if os.path.isfile(chemin):
            donnees = pd.read_csv(chemin)
            return donnees
        else:
            print("chemin invalide")
            return

    """Cette fonction permet de séparer les données d'entrées des cibles du fichier csv(d'entrainement ou de test)"""

    def encoder(self, dEntrainement, dTest):
        cibles = dEntrainement.species
        labelEncoder = LabelEncoder().fit(cibles)
        self.Y_entrainement = labelEncoder.transform(cibles)
        self.classes = list(labelEncoder.classes_)
        self.idTests = dTest['id']

        self.X_entrainement = dEntrainement.drop(['species', 'id'], axis=1)
        self.X_test = dTest.drop("id", axis=1)

        return self.X_entrainement, self.Y_entrainement, self.X_test

    def normaliser(self, dEntrainement):
        
