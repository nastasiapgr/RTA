# import libraries and data;

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import neighbors
from flask import request
from flask import jsonify
from sklearn import datasets

import pickle
from flask import Flask, request, jsonify, render_template
import joblib

iris = datasets.load_iris()

# creating variables x,y as data and target
x = iris.data
y = iris.target

# dividing dataset into train and test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)


class Perceptron:

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)


model = Perceptron()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])

X, y = df.iloc[:100, [0, 2]].values, df.iloc[:100, 4]


def a(x):
    if x == 0:
        return -1
    else:
        return 1


y = y.map(a)

# loading library
import pickle
# create an iterator object with write permission - model.pkl
with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)

# load saved model
with open('model_pkl' , 'rb') as f:
    model = pickle.load(f)

model.fit(X, y)

model.predict([3.4, 3.1])


def score(sample):
    import numpy as np
    np_sample = np.array(sample)
    pred = model.predict(np_sample.reshape(1, -1)).tolist()
    return ['setosa', 'versicolor', 'virginica'][pred[0]]

# Create a flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_prediction():

    # sepal length
    sepal_length = request.args.get('sl')
    # sepal width
    sepal_width = request.args.get('sw')
    # petal length
    petal_length = request.args.get('pl')
    # petal width
    petal_width = request.args.get('pw')

    # The features of the observation to predict
    features = [sepal_length,
                sepal_width,
                petal_length,
                petal_width]

    # Predict the class using the model
    predicted_class = int(model.predict([features]))

    # Return a json object containing the features and prediction
    return render_template( 'index.html', prediction_text='The Flower is {}'.format(predicted_class))
if __name__ == '__main__':
    # Run the app at 0.0.0.0:3333
    app.run(port=3333,host='0.0.0.0', debug=True)