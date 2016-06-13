#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Classify data using scikit-learn
# And calculate confusion matrices

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib import rc
import scipy.optimize as optimization
from sklearn import neighbors, linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.cross_validation import train_test_split

# Plot style
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.style.use('bmh')

# Import data from file
datos_file = 'datos_clasificacion.dat'
datos = np.loadtxt(datos_file)

labels = ['$\mathrm{Clase}\,1$', '$\mathrm{Clase}\,2$']

x1 = datos[:, 0]
x2 = datos[:, 1]
class_label = datos[:, 2]

# Get filters for different classes
filter_1 = class_label == 1
filter_2 = class_label == 2

# Plot data without classification
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_pre.pdf')
plt.show()

n_neighbors = 20  # Number of neighbors for NN method
X = np.transpose((x1, x2))
h = .2  # Mesh size for plotting decision boundaries

# Nearest Neighbors
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
clf.fit(X, class_label)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Prediction

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.3, cmap='coolwarm')

plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best', title='$\mathrm{Nearest-Neighbors}$')
# plt.savefig('T5_p1_b_NN.pdf')
plt.show()
	
# AdaBoost
clf = AdaBoostClassifier(n_estimators=100)
clf.fit(X, class_label)  # Train classificator
	
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Prediction

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.3, cmap='coolwarm')
	
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
# plt.legend(loc='best', title='$\mathrm{AdaBoost}$')
plt.savefig('T5_p1_b_ada.pdf')
plt.show()


# Linear Regression
clf = linear_model.LinearRegression()
clf.fit(X, class_label)  # Train Classificator
	
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # Prediction

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.3, cmap='coolwarm')
	
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best', title='$\mathrm{Linear\,Regression}$')
# plt.savefig('T5_p1_pre.pdf')
plt.show()


# LDA
clf = LinearDiscriminantAnalysis()
clf.fit(X, class_label)
	
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.3, cmap='coolwarm')
	
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best', title='$\mathrm{LDA}$')
# plt.savefig('T5_p1_pre.pdf')
plt.show()


# QDA
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, class_label)
	
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, alpha=0.3, cmap='coolwarm')
	
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best', title='$\mathrm{QDA}$')
# plt.savefig('T5_p1_pre.pdf')
plt.show()


cm_1 = np.zeros((2,2))
cm_2 = np.zeros((2,2))
cm_3 = np.zeros((2,2))
cm_4 = np.zeros((2,2))
cm_5 = np.zeros((2,2))

as_1 = 0.0
as_2 = 0.0
as_3 = 0.0
as_4 = 0.0
as_5 = 0.0

max_value = 300

for random_state in np.arange(1, max_value):
	# Confusion Matrix
	# First, split data in training and validation sets
	X_train, X_test, y_train, y_test = train_test_split(X, class_label, random_state=random_state)

	# Initialize Classificators
	clf_1 = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
	clf_2 = AdaBoostClassifier(n_estimators=100)
	clf_3 = linear_model.LinearRegression()
	clf_4 = LinearDiscriminantAnalysis()
	clf_5 = QuadraticDiscriminantAnalysis()

	# Test classificator with training and validation data
	y_pred_1 = clf_1.fit(X_train, y_train).predict(X_test)
	y_pred_2 = clf_2.fit(X_train, y_train).predict(X_test)
	y_pred_3 = clf_3.fit(X_train, y_train).predict(X_test)
	y_pred_4 = clf_4.fit(X_train, y_train).predict(X_test)
	y_pred_5 = clf_5.fit(X_train, y_train).predict(X_test)

	'''
	Z_1 = clf_1.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z_2 = clf_2.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z_3 = clf_3.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z_3 = Z_3.reshape(xx.shape)
	Z_4 = clf_4.decision_function(np.c_[xx.ravel(), yy.ravel()])
	Z_5 = clf_5.decision_function(np.c_[xx.ravel(), yy.ravel()])
	'''

	# Obtain confusion matrices
	cm_1 += confusion_matrix(y_test, y_pred_1)
	cm_2 += confusion_matrix(y_test, y_pred_2)
	# cm_3 += confusion_matrix(y_test, y_pred_3)
	cm_4 += confusion_matrix(y_test, y_pred_4)
	cm_5 += confusion_matrix(y_test, y_pred_5)

	# Misclassification rate
	as_1 += zero_one_loss(y_test, y_pred_1)
	as_2 += zero_one_loss(y_test, y_pred_2)
	# as_3 += zero_one_loss(y_test, y_pred_3)
	as_4 += zero_one_loss(y_test, y_pred_4)
	as_5 += zero_one_loss(y_test, y_pred_5)


size_set = np.shape(np.arange(1, max_value))[0]

# Print confusion matrices
print(cm_1 / size_set)
print(cm_2 / size_set)
# print(cm_3 / size_set)
print(cm_4 / size_set)
print(cm_5 / size_set)


# Print misclassification rates
print(as_1 / size_set)
print(as_2 / size_set)
# print(as_3 / size_set)
print(as_4 / size_set)
print(as_5 / size_set)

