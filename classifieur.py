from importlib import import_module
#from sklearn.svm import SVC
from data_management import gestion_donnees as gd
from sklearn.metrics import accuracy_score

class Classifieur:
    def __init__(self, classModule, className, params={}):
        self.classModule = classModule
        self.className = className
        self.params = params
        self.classifieur = None

    def create(self):
        module = import_module(self.classModule)
        class_ = getattr(module, self.className)
        self.classifieur = class_()
        self.classifieur.set_params(**self.params)

    def entraine(self, X, Y):
        self.classifieur.fit(X,Y)

    def prediction(self, X):
        predictions = self.classifieur.predict(X)
        return predictions
