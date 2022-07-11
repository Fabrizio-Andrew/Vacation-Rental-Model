from sagemaker.sklearn.model import SKLearnModel
import boto3
import sagemaker
import os

os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

sagemaker.Session(boto3.session.Session())

role = 'AmazonSageMakerServiceCatalogProductsUseRole'
aws_sklearn = SKLearnModel(entry_point='serve_model.py',
                        model_data='s3://sagemaker-studio-efge0ycc03/model.tar.gz',
                        role=role,
                        framework_version='0.20.0',
                        py_version='py3')

aws_sklearn_model = aws_sklearn.deploy(instance_type='ml.m4.xlarge',
                                            initial_instance_count=1)

print(aws_sklearn_model.endpoint)