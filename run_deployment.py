from pipelines.deployment_pipeline import deployment_pipeline, inference_pipeline
import click

from pipelines.deployment_pipeline import continuous_deployment_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


DEPLOY = "DEPLOY"
PREDICT = "PREDICT"
DEPLOY_AND_PREDICT = "DEPLOY_AND_PREDICT"
@click.command()
@click.option(
    "--config",
    "--c", 
    type = click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default= "DEPLOY_AND_PREDICT",
    help= """optionally you can choose to only run the deployment pipeline to train and 
    deploy a model ('deploy'), or to only run a prediction against the deployed model ('predict'), 
    by default both will be run ('deploy_and_predict')."""
)

@click.option(
    "--min-accuracy",
    default = 0.92,
    help="minimum accuracy required to deploy the model"
)
def run_deployment(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continuous_deployment_pipeline(
            mon_accuracy = min_accuracy,
            workers = 3,
            timeout= 60)
    if predict:
        inference_pipeline()