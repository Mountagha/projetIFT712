import pandas as pd
from sklearn.svm import SVC
from data_management import gestion_donnees as gd
from sklearn.metrics import accuracy_score
from classifieur import Classifieur
import warnings

#classifiers = [SVC(kernel="rbf", C=0.02, probability=True)]

gd = gd()
train = gd.chargement_donnees("train.csv")
test = gd.chargement_donnees("test.csv")

#train, train, = gd.encoder(train, test)
X, Y = gd.encoder(train)
x_train, y_train, x_test, y_test = gd.split(X,Y)
clf = Classifieur("sklearn.svm", "SVC", params={"kernel":"rbf", "C":0.02, "probability":True})
#clf = Classifieur("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis", params={})
clf.create()
clf.entraine(x_train, y_train)
train_predictions = clf.prediction(x_test)
acc = accuracy_score(y_test, train_predictions)
print("accuracy_score : {:.4%}".format(acc))

#X, Y, X_test, ids, classes = gd.encoder(train, test)
#print(len(classes))
#fClf = Classifieur("sklearn.discriminant_analysis", "LinearDiscriminantAnalysis", params={})
#fClf.create()
#fClf.entraine(X, Y)
#predictions = clf.prediction(X_test)
#acc = accuracy_score(Y, predictions)
#print("accuracy_score : {:.4%}".format(acc))
#submission = pd.DataFrame(predictions, columns=classes)
#submission.insert(0, 'id', ids)
#submmission.reset_index()
#submission.tail()
