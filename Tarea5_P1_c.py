#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Obtain bayes classificator
# Using real parameters from data points

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib import rc
import scipy.optimize as optimization
import matplotlib.mlab as mlab
from sklearn import neighbors, linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
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

plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_pre.pdf')
plt.show()

# Real parameters

mu_1 = np.array([2., 3.])
mu_2 = np.array([6., 6.])

cov_1 = np.array([[5., -2.], [-2., 5.]])
cov_2 = np.array([[1., 0.], [0., 1.]])

x = np.linspace(-6.0, 10, 100)
y = np.linspace(-6.0, 10, 100)
X, Y = np.meshgrid(x, y)

Z1 = mlab.bivariate_normal(X, Y, sigmax=cov_1[0, 0], sigmay=cov_1[1, 1], mux=mu_1[0], muy=mu_1[1], sigmaxy=cov_1[1, 0])
Z2 = mlab.bivariate_normal(X, Y, sigmax=cov_2[0, 0], sigmay=cov_2[1, 1], mux=mu_2[0], muy=mu_2[1], sigmaxy=cov_2[1, 0])

Z = (Z2 - Z1)

plt.figure()
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
# plt.contour(X, Y, Z1)
# plt.contour(X, Y, Z2)
CS = plt.contour(X, Y, Z, [0])
CS.collections[0].set_label('$\mathrm{Bayes\,Classification}$')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_c_zero.pdf')
plt.show()

# Separate contours
plt.figure()
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.contour(X, Y, Z1)
plt.contour(X, Y, Z2)
# CS.collections[0].set_label('$\mathrm{Bayes\,Classification}$')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_c_both.pdf')
plt.show()
