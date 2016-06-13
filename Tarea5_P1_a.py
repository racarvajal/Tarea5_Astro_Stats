#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use linear regression, LDA and QDA
# to classify data

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from matplotlib import rc
import scipy.optimize as optimization

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

# Plot original data
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_pre.pdf')
plt.show()

# Implement Linear Regression

def multilin(params, xdata, ydata):
	return (ydata - (params[0] + params[1] * xdata[0] + params[2] * xdata[1]))

x_0 = (0, 0, 0)  # First guess for linear regression
params_result = optimization.leastsq(multilin, x_0, args=((x1, x2), class_label))

print(params_result)  # Results

# Boundary (Y = 1.5)
x2_line = (1.5 - params_result[0][0] - params_result[0][1] * x1) / params_result[0][2]

# Plot data with linear regression classification
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.plot(x1, x2_line, label='$\mathrm{Linear\,Regression}$')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_a_LR.pdf')
plt.show()

# Sizes of classes
N_1 = np.shape(x1[filter_1])[0]
N_2 = np.shape(x1[filter_2])[0]

# LDA
p_1 = float(N_1) / float((np.shape(x1)[0]))  # Estimated Prior for class 1
p_2 = float(N_2) / float((np.shape(x1)[0]))  # Estimated Prior for class 2

mu_1 = np.array([sum(x1[filter_1]) / N_1, sum(x2[filter_1]) / N_1])
mu_2 = np.array([sum(x1[filter_2]) / N_2, sum(x2[filter_2]) / N_2])


difference_1 = np.array([x1[filter_1], x2[filter_1]]).reshape((-1, 2)) - np.repeat(mu_1, N_1).reshape((-1, 2))
difference_2 = np.array([x1[filter_2], x2[filter_2]]).reshape((-1, 2)) - np.repeat(mu_2, N_2).reshape((-1, 2))

diff_dot_diff_1 = np.dot(np.transpose(difference_1), difference_1)
diff_dot_diff_2 = np.dot(np.transpose(difference_2), difference_2)

sigma_both = (diff_dot_diff_1 + diff_dot_diff_2) / (N_1 + N_2 - 2)  # Estimated covariance matrix


# LDA classifiers
# Decision boundary
A = (mu_1 + mu_2)
B = (mu_1 - mu_2)
sigma_inv = inv(sigma_both)
sigma_B = np.dot(sigma_inv, B)

# Boundary (delta_1 = delta_2)
x2_lda = (0.5 * np.dot(np.transpose(A), sigma_B) - np.log(p_1 / p_2) - x1 * sigma_B[0]) / sigma_B[1]

# Plot data with linear discriminant analysis classification
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
# plt.plot(x1, x2_line, label='$\mathrm{Linear\,Regression}$')
plt.plot(x1, x2_lda, label='$\mathrm{LDA\,Regression}$')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_a_LDA.pdf')
plt.show()

# QDA
sigma_1 = diff_dot_diff_1 / (N_1 - 1)
sigma_2 = diff_dot_diff_2 / (N_2 - 1)
sigma_1_inv = inv(sigma_1)
sigma_2_inv = inv(sigma_2)
sigma_inv_diff = sigma_1_inv - sigma_2_inv
sigma_mu_1 = np.dot(sigma_1_inv, mu_1)
sigma_mu_2 = np.dot(sigma_2_inv, mu_2)
sigma_mu_diff = sigma_mu_1 - sigma_mu_2
mu_sigma_mu_1 = np.dot(np.transpose(mu_1), sigma_mu_1)
mu_sigma_mu_2 = np.dot(np.transpose(mu_2), sigma_mu_2)

# Boundary (delta_1 = delta_2)
a = sigma_inv_diff[1, 1] * 0.5
b = (x1 * 0.5 *(sigma_inv_diff[0, 1] + sigma_inv_diff[1, 0]) - sigma_mu_diff[1])
c = np.multiply(x1, x1) * sigma_inv_diff[0, 0] * 0.5 - x1 * sigma_mu_diff[0] + 0.5 * (np.dot(np.transpose(mu_1), sigma_mu_1) - np.dot(np.transpose(mu_2), sigma_mu_2)) - np.log(p_1 / p_2) + 0.5 * np.log(np.linalg.det(sigma_1) / np.linalg.det(sigma_2))
d = np.multiply(b, b) - 4 * np.multiply(np.repeat(a, N_1 + N_2), c)

# x2_qda = (-b + np.sqrt(d)) / (2 * a)
x = np.linspace(-6.0, 10, 100)
y = np.linspace(-6.0, 10, 100)
X, Y = np.meshgrid(x, y)
x2_qda = a * Y**2 + (X * 0.5 *(sigma_inv_diff[0, 1] + sigma_inv_diff[1, 0]) - sigma_mu_diff[1]) * Y + np.multiply(X, X) * sigma_inv_diff[0, 0] * 0.5 - X * sigma_mu_diff[0] + 0.5 * (np.dot(np.transpose(mu_1), sigma_mu_1) - np.dot(np.transpose(mu_2), sigma_mu_2)) - np.log(p_1 / p_2) + 0.5 * np.log(np.linalg.det(sigma_1) / np.linalg.det(sigma_2))

# Plot data with quadratic discriminant analysis classification
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
# plt.plot(x1, x2_line, label='$\mathrm{Linear\,Regression}$')
# plt.plot(x1, x2_lda, label='$\mathrm{LDA\,Regression}$')
CS = plt.contour(X, Y, x2_qda, [0], label='$\mathrm{QDA\,Regression}$')
CS.collections[0].set_label('$\mathrm{QDA\,Regression}$')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_a_QDA.pdf')
plt.show()

# Plot data with three classification methods
plt.scatter(x1[filter_1], x2[filter_1], s=class_label[filter_1]*15, c='b', marker='p', alpha=1.0, label=labels[0])
plt.scatter(x1[filter_2], x2[filter_2], s=class_label[filter_2]*15, c='brown', marker='p', alpha=1.0, label=labels[1])
plt.plot(x1, x2_line, label='$\mathrm{Linear\,Regression}$')
plt.plot(x1, x2_lda, label='$\mathrm{LDA\,Regression}$')
CS = plt.contour(X, Y, x2_qda, [0], label='$\mathrm{QDA\,Regression}$')
CS.collections[0].set_label('$\mathrm{QDA\,Regression}$')
plt.xlabel('$x_{1}$')
plt.ylabel('$x_{2}$')
plt.legend(loc='best')
# plt.savefig('T5_p1_a_all.pdf')
plt.show()

