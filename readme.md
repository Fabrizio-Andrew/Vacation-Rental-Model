# Vacation Rental Model Build/Deploy Scripts

This repository contains a set of scripts to deploy a basic linear model to AWS SageMaker.
It is primarily based on the guide at: https://towardsdatascience.com/deploying-a-pre-trained-sklearn-model-on-amazon-sagemaker-826a2b5ac0b6

### Prerequisites
* Install Python 3 + pip

* Create a virtualenv and install dependencies
```bash
# Create virtual environment
$ python venv venv
# Install Dependencies
$ pip install -r requirements.txt
```

This script also assumes the user has set up an AWS account and installed the AWS CLI.

### Set up & Deploy the model to SageMaker
* Create/update the joblib file (if needed)
```bash
$ python local_model.py
```

* Deploy the model to SageMaker
```bash
$ python deploy_model.py
```