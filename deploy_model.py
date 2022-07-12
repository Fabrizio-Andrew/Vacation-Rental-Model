### See https://towardsdatascience.com/deploying-a-pre-trained-sklearn-model-on-amazon-sagemaker-826a2b5ac0b6

import boto3
import sagemaker
import time
from time import gmtime, strftime
import os
import joblib
import pickle
import tarfile
from sagemaker.estimator import Estimator
import subprocess

# Setup
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
print('REGION: ' + region)
sagemaker_session = sagemaker.Session(boto_session)
role = 'arn:aws:iam::653988117753:role/service-role/AmazonSageMaker-ExecutionRole-20220701T203623'

# Build tar file
bashCommand = 'tar -cvpzf model.tar.gz model.joblib serve_model.py'
process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
output, error = process.communicate()

# Set up S3 Bucket
default_bucket = sagemaker_session.default_bucket()
print(default_bucket)

# upload tar.gz to bucket
model_artifact_uri = f's3://{default_bucket}/model.tar.gz'
response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'model.tar.gz')

# Retrieve sklearn image
image_uri = sagemaker.image_uris.retrieve(
    framework='sklearn',
    region=region,
    version='0.23-1',
    py_version='py3',
    instance_type='ml.m5.xlarge',
)

# Model Creation
model_name = 'sklearn-model' + strftime('%Y-%m-%d-%H-%M-%S', gmtime())
print('MODEL NAME: ' + model_name)
create_model_response = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            'Image': image_uri,
            'Mode': 'SingleModel',
            'ModelDataUrl': model_artifact_uri,
            'Environment': {'SAGEMAKER_SUBMIT_DIRECTORY': model_artifact_uri,
                            'SAGEMAKER_PROGRAM': 'serve_model.py'}
        }
    ],
    ExecutionRoleArn=role
)
print('MODEL ARN: ' + create_model_response['ModelArn'])


# Endpoint Config Creation
sklearn_epc_name = 'sklearn-epc' + strftime('%Y-%m-%d-%H-%M-%S', gmtime())
endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName=sklearn_epc_name,
    ProductionVariants=[
        {
            'VariantName': 'sklearnvariant',
            'ModelName': model_name,
            'InstanceType': 'ml.c5.large',
            'InitialInstanceCount': 1
        }
    ]
)
print('ENDPOINT CONFIGURATION ARN: ' + endpoint_config_response['EndpointConfigArn'])

# Endpoint Creation
endpoint_name = 'sklearn-ep' + strftime('%Y-%m-%d-%H-%M-%S', gmtime())
create_endpoint_response = client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=sklearn_epc_name
)
print('ENDPOINT ARN: ' + create_endpoint_response['EndpointArn'])

# Monitor deployment
describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
while describe_endpoint_response['EndpointStatus'] == 'Creating':
    describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
    print(describe_endpoint_response['EndpointStatus'])
    time.sleep(15)
print(describe_endpoint_response)
