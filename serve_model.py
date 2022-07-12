import argparse
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # Create parser object to collect environment variables from container
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    args = parser.parse_args()

# This is where I would load data and train if I had data...

# Set the model via coefs and intercept (Instead of training it)
lm = LinearRegression()
lm.intercept_ = 0.12633143197779734
lm.coef_ = [-0.18180501, -0.32107893, -0.36701204, -0.15250934, -0.05541475,
    -0.03938244,  0.55789099,  0.01080494,  0.10483031,  0.25154944,
        0.39175801]

# Save the model to location specified by args.model_dir
with open('model.joblib', 'wb') as f:
    joblib.dump(lm,f)

def model_fn(model_dir):
    """
    Loads the model that was saved at the end of the __main__ block to be used
    by the predict_fn function below.
    """
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))

    return model

def input_fn(request_body, request_content_type):
    """
    Formats a request body as a numpy array that is sent to the deployed model.
    """
    if request_content_type == 'text/csv':
        samples = []
        for r in request_body.split('|'):
            samples.append(list(map(float,r.split(','))))
        return np.array(samples)
    else:
        raise ValueError('This model only supports text/csv input.')


def predict_fn(input_data, model):
    """
    Makes a prediction on the data formatted by input_fn.
    """
    return model.predict(input_data)

#def output_fn(prediction, content_type):
    """
    Reformats prediction result from predict_fn to create the API call response.
    """
#    return '|'.join([INDEX_TO_LABEL[t] for t in prediction])