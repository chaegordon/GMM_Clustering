# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:34:25 2019

@author: chaeg

Pythia Sports Data Analysis Excercise
"""
#importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


#import sci-kit learn module

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_spd_matrix
from sklearn.utils import shuffle

#defining N, the number of GMM in the model
N=4

""" Need to insert your own csv data where it says sampledata_20190313.csv it should be only 2 columns"""


#creating pandas data frame from the given csv data
column_headers = ['x','y']
df = pd.read_csv('sampledata_20190313.csv', names = column_headers)
print(df.head())#looking at the form of the data 

#plot x and y against the column number
df.plot()

#plot one column against the other as a scatter plot
df.plot.scatter(x='x', y='y')

#creating the model
clf = GaussianMixture(n_components=4, covariance_type='full')

#fitting the model to the data
clf.fit(df)

#predicting cluster lables

labels = clf.predict(df)

#plotting the data with the clusters labelled
df.plot.scatter(x = 'x', y= 'y', c=labels, cmap='viridis')

#getting attributes of the model
means = clf.means_
covariance = clf.covariances_

mean = pd.DataFrame(means, columns = ['x_mean','y_mean'])

covar_model_1 = covariance[0,:,:]
covar_model_2 = covariance[1,:,:]
covar_model_3 = covariance[2,:,:]
covar_model_4 = covariance[3,:,:]

#prints the coordinates of the means in matrix form
print("The mean matrix is:")
print(means)

time.sleep(0.1)
#prints out the covariance matrices
for i in range(3):
  count = i
  print("The covariance marix for the " + str(count + 1) + " gaussian:")
  print(covariance[i,:,:])

#sets number of data points
number_samples = 1000

counter_x = np.zeros(number_samples)
counter_y = np.zeros(number_samples)
counter_z = np.zeros(number_samples)

#create a sample distribution of N 3D GMM
for i in range(N):
    centre = np.random.rand(3)
    cov_mat = make_spd_matrix(n_dim = 3, random_state=None)
    X, Y, Z = np.random.multivariate_normal(mean = centre, cov = cov_mat, size = (number_samples)).T
    counter_x += X
    counter_y += Y
    counter_z += Y

#shuffle the data so its not already grouped
counter_x = shuffle(counter_x)
counter_y = shuffle(counter_y)
counter_z = shuffle(counter_z)

#create data frame of the generated samples
d = {'x' : counter_x, 'y' : counter_y, 'z' : counter_z}
gen_df = pd.DataFrame(data = d)

#plot the 3D generated samples
threedee = plt.figure().gca(projection='3d')
threedee.scatter(gen_df.x, gen_df.y, gen_df.z)
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
plt.show()

#now retry making a model to predict the GMM

clf_3 = GaussianMixture(n_components=4, covariance_type='full')

#fitting the model to the data
clf_3.fit(gen_df)

#predicting cluster lables

labels_3 = clf_3.predict(gen_df)

#apply similar method of adding color labels
threedee = plt.figure().gca(projection='3d')
threedee.scatter(gen_df.x, gen_df.y, gen_df.z, c=labels_3, cmap='viridis')
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
plt.show()
