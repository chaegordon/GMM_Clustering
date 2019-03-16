# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:02:45 2019

@author: chaeg
"""

#importing necessary modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


#import sci-kit learn module

from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_spd_matrix
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

#defining N, the number of GMM in the model
N=4

#defining number of samples in generated the data sets
number_samples = 2000

#creating pandas data frame from the given csv data
column_headers = ['x','y']
df = pd.read_csv(r'C:\Users\chaeg\OneDrive\Documents\Internships\Pythia_Excercise\sampledata_20190313.csv', names = column_headers)

#print(df.head())#looking at the form of the data 

#plot x and y against the column number
df.plot()

#plot one column against the other as a scatter plot
ax = df.plot.scatter(x='x', y='y')
fig = ax.get_figure()
fig.suptitle('Data Provided for Exercise', fontsize=16)
fig.savefig('Q1.png')


#creating the model
clf = GaussianMixture(n_components=4, covariance_type='full')

#fitting the model to the data
clf.fit(df)

#predicting cluster lables

labels = clf.predict(df)

#plotting the data with the clusters labelled
ax_1 = df.plot.scatter(x = 'x', y= 'y', c=labels, cmap='viridis')
fig_1 = ax_1.get_figure()
fig_1.suptitle('Clustered Data', fontsize=16)
fig_1.savefig('Q2.png')

#getting attributes of the model
means = clf.means_
covariance = clf.covariances_

mean = pd.DataFrame(means, columns = ['x_mean','y_mean'])

#this is just for convenience so I remember the slicing I want to use in the for loop
covar_model_1 = covariance[0,:,:]
covar_model_2 = covariance[1,:,:]
covar_model_3 = covariance[2,:,:]
covar_model_4 = covariance[3,:,:]

#prints the coordinates of the means in matrix form
print("The mean matrix is:")
print(means)

time.sleep(0.1)
#prints out the covariance matrices
for i in range(4):
  count = i
  print("The covariance marix for the " + str(count + 1) + " gaussian:")
  print(covariance[i,:,:])

#get a handle of the goodness of fit of the model
#Including the silhouette score
print("The silhouette score is "+ str(silhouette_score(df, labels, metric = 'euclidean')))


#for comparison with a different algorithm
print("The 2D model had a BIC " + str(clf.bic(df)))
print("and the average log likelihood per sample is " + str(clf.score(df)))


