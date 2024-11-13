from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math

from perceptron import Perceptron

iris_dataset = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.3, random_state=5)

perceptron = Perceptron()
perceptron.train(X_train, y_train, n_iter=100, learning_rate=0.03)

y_predicted = perceptron.predict(X_test)
y_predicted = [math.ceil(y) for y in y_predicted]
accuracy = accuracy_score(y_test, y_predicted)

print(f'Accuracy of our percetron {round(accuracy,3)*100}%')
