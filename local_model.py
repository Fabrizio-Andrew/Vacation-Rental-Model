from sklearn.linear_model import LinearRegression
import joblib
import numpy as np


model = LinearRegression()
model.intercept_ = 0.12633143197779734
model.coef_ = [-0.18180501, -0.32107893, -0.36701204, -0.15250934, -0.05541475,
    -0.03938244,  0.55789099,  0.01080494,  0.10483031,  0.25154944,
        0.39175801]

with open('model.joblib', 'wb') as f:
    joblib.dump(model,f)


with open('model.joblib', 'rb') as f:
    predictor = joblib.load(f)

print("Testing following input: ")
sampInput = np.array([[0., 0.76013424, 0., 0., 0.45095014, -1.09222555, 0., 0.7678654, 0., 0.15412874, -0.57417196]])
print(type(sampInput))
print('========PREDICTION========')
print(predictor.predict(sampInput))