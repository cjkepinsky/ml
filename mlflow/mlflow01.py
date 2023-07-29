import mlflow

if __name__ == "__name__":
    mlflow.set_experiment(experiment_name='exp_test1')
    with mlflow.start_run():
        mlflow.log_param('b1', 2)
        mlflow.log_metric('m1', 1)

