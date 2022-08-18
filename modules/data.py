from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class Transform:
    def __init__(self) -> None:
        pass
    def fit(self,X_train):
        self.scale.fit(X_train)

    def normalize(self,X_train,X_test):
        self.scale = StandardScaler()
        self.fit(X_train)

        return self.scale.transform(X_train),self.scale.transform(X_test)
    
    def split(self,X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        return X_train, X_test, y_train, y_test

class Data:
    def __init__(self) -> None:
        self.transform = Transform()
    
    def generate(self):
        X, y = make_classification(
                n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
            )
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        datasets = [
                make_moons(noise=0.2, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable,
            ]
        return datasets

