from sklearn.svm import SVC
from data_management import gestion_donnees as gd
from sklearn.metrics import accuracy_score

class Classifieur:
    def __init__(self, className, params={}):
        self.className = className
        self.params = params
        self.classifieur = None

    def create(self):
        if self.className == 'SVC':
            print("choix de svc")
            self.classifieur = SVC()
            self.classifieur.set_params(**self.params)

    def entraine(self, X, Y):
        self.classifieur.fit(X,Y)
