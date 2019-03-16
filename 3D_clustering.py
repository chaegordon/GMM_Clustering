# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:13:19 2019

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
N=5

#defining number of samples in generated the data sets
number_samples = 2000

# make 4-class 3D dataset for classification

transformation = transformation = np.random.rand(3,N)

#make a random standard deviation matrix
clust_st = []
for i in range(N):
    clust_st.append(np.random.rand(1))
    
    

Ex, whY = make_blobs(n_samples= number_samples,cluster_std = clust_st , centers= N, random_state=40, n_features = 3)

#here Ex contains the sample blobs (x,y,z) and whY contains the labels for the clusters


print("The blobs are created in the shape " + str(Ex.shape))

Ex_aniso = np.dot(Ex, transformation)

blob_df = pd.DataFrame(data = Ex, columns = ['x','y','z'])

threedee = plt.figure().gca(projection='3d')
#blob_df.x, blob_df.y, blob_df.z
#Ex[:,0], Ex[:,1], Ex[:,2]
threedee.scatter(blob_df.x, blob_df.y, blob_df.z)
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
fig_1 = threedee.get_figure()
fig_1.suptitle('3D Data', fontsize=16)
fig_1.savefig('Q6_4.png')
plt.show()

clf_blob = GaussianMixture(n_components=N, covariance_type='full')

#fitting the model to the data
clf_blob.fit(blob_df)

#predicting cluster lables

labels_blob = clf_blob.predict(blob_df)

#plotting the labelled distribution as before
threedee = plt.figure().gca(projection='3d')
threedee.scatter(blob_df.x, blob_df.y, blob_df.z, c=labels_blob, cmap='viridis')
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
fig_2 = threedee.get_figure()
fig_2.suptitle('3D Clustered Data', fontsize=16)
fig_2.savefig('Q6_5.png')
plt.show()

print("The silhouette score is "+ str(silhouette_score(blob_df, labels_blob, metric = 'euclidean')))

print("the blob model BIC is " + str(clf_blob.bic(blob_df)))

#this is related to the error in labelling although because there is more than
#2 clusters it wont be 1 to 1

error_measure_blob = sum(whY - labels_blob)

print('error measure of blob is ' + str(error_measure_blob))

if error_measure_blob < 1:
    print("model fits well")
else:
    print("model doesn't fit well")
