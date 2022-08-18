from sklearn.metrics import f1_score,precision_score,recall_score ,accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

class Validate:
    def __init__(self) -> None:
        pass

    def eval(self,model,X_test,X_train,y_train,y_test):

        self.cv   = StratifiedKFold(n_splits=10, random_state=41, shuffle=True)        
        cv_result = cross_val_score(model,X_train,y_train, cv=self.cv, n_jobs=-1)

        y_hat_test   = model.predict(X_test)
        y_hat_train  = model.predict(X_train)

        metrics = {
            "10-fold accuracy":      cv_result.mean().round(3),
            "10-fold deviation":     cv_result.std().round(3),

            "train accuracy_score":  accuracy_score(  y_train, y_hat_train ).round(3),
            "train f1_score":        f1_score(        y_train, y_hat_train ).round(3),
            "train precision_score": precision_score( y_train, y_hat_train ).round(3),
            "train recall_score":    recall_score(    y_train, y_hat_train ).round(3),

            "test accuracy_score":   accuracy_score(  y_test, y_hat_test ).round(3),
            "test f1_score":         f1_score(        y_test, y_hat_test ).round(3),
            "test precision_score":  precision_score( y_test, y_hat_test ).round(3),
            "test recall_score":     recall_score(    y_test, y_hat_test ).round(3)
        }

        return metrics 