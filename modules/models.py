from xgboost          import XGBClassifier
from sklearn.tree     import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier



class Models:
    def __init__(self) -> None:
        
        self.emsemble_names = ["AdaBoostClassifier","BaggingClassifier"]

        self.models = {
            "DecisionTreeClassifier":     DecisionTreeClassifier,
            "AdaBoostClassifier":         AdaBoostClassifier,
            "BaggingClassifier":          BaggingClassifier,
            "RandomForestClassifier":     RandomForestClassifier,
            "ExtraTreesClassifier":       ExtraTreesClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "XGBClassifier":              XGBClassifier
        }
    
    def instantiate(self,model_name,parameter):
        if model_name in self.emsemble_names:
        
            p = parameter[1]
            
            base_estimator = self.models["DecisionTreeClassifier"](**p)
            parameter[0]["base_estimator"] = base_estimator

            model = self.models[model_name](**parameter[0])
            
        else:
            
            model = self.models[model_name](**parameter)

        return model
