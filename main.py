from sklearn.svm import SVC
from data_management import gestion_donnees as gd
from sklearn.metrics import accuracy_score
from classifieur import Classifieur

#classifiers = [SVC(kernel="rbf", C=0.02, probability=True)]

gd = gd()
train = gd.chargement_donnees("train.csv")
test = gd.chargement_donnees("test.csv")

x_train, y_train, x_test = gd.encoder(train, test)
clf = Classifieur("SVC", params={"kernel":"rbf", "C":0.02, "probability":True})
clf.create()
clf.entraine(x_train, y_train)
