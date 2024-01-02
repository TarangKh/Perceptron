import numpy as np 
import pandas as pd 
import joblib
import os

# Perceptron classd to solve AND gate and OR gate
class Perceptron:
    def __init__(self, eta, epochs):
        self.eta = eta
        self.epochs = epochs
        self.weights = np.random.randn(3) * 1e-4

    def activationFunction(self, input, weights):
        z = np.dot(input, weights)
        return np.where(z > 0, 1, 0)
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        # adding bias to the training data
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]

        for epoch in range(self.epochs):
            y_pred = self.activationFunction(X_with_bias, self.weights)
            error = self.y - y_pred
            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, error)


    def predict(self, X):
        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        return self.activationFunction(X_with_bias, self.weights)




# Testing on AND gate

data = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 0, 0, 1]
}

AND = pd.DataFrame(data)
X = AND[["x1", "x2"]]
y = AND["y"]

eta = 0.5
epochs = 10
model = Perceptron(eta, epochs)
model.fit(X, y)
output = model.predict(X)
print(output)

dir = "Perceptron Model"
os.makedirs(dir, exist_ok=True)
filename = os.path.join(dir, "ANDgate.model")
joblib.dump(model, filename)

# Testing on OR Gate
data = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 1, 1, 1]
}

AND = pd.DataFrame(data)
X = AND[["x1", "x2"]]
y = AND["y"]
model.fit(X, y)
output = model.predict(X)
print(output)

filename = os.path.join(dir, "ORgate.model")
joblib.dump(model, filename)

# Testing on XOR Gate
data = {
    "x1": [0, 0, 1, 1],
    "x2": [0, 1, 0, 1],
    "y": [0, 1, 1, 0]
}

AND = pd.DataFrame(data)
X = AND[["x1", "x2"]]
y = AND["y"]
model.fit(X, y)
output = model.predict(X)
print(output)


# Conclusion 
"""
    The perceptron is successfully able to solve AND and OR gate but failed to predict the correct output
    of XOR gate. Hence, we can see the perceptron can only classify linear problem like AND and OR gate 
    but in the case of XOR gate it failed to classify because it is a non-linear problem.
"""