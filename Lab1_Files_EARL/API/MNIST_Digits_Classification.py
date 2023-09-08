import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib


mnist = load_digits(return_X_y=False)

y = pd.DataFrame(mnist.target)

x = pd.DataFrame(mnist.data)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2,
                                                random_state=42)


from sklearn.linear_model import SGDClassifier
model = SGDClassifier()
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)

accuracy = accuracy_score(ytest, y_pred)
print(f"Accuracy test: {accuracy * 100:.2f}%")

#Save the trained model
joblib.dump(model,'mnist_classification_model.plk')

