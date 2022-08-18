
import mlflow
import time

class MlFlow:
    def __init__(self) -> None:
        mlflow.set_experiment(experiment_name ='aula_GPS')
        
    def log_result(self,paramters,metrics):

        with mlflow.start_run(run_name = paramters["model_name"]+"-"+str(time.time())):
            print(paramters)
            mlflow.log_params(paramters)
            mlflow.log_metrics(metrics)