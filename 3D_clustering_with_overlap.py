# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:08:41 2019

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
N=2

#defining number of samples in generated the data sets
number_samples = 2000


counter_x = np.zeros(number_samples)
counter_y = np.zeros(number_samples)
counter_z = np.zeros(number_samples)


counter_mean = []
counter_cov_mat = []

#create a sample distribution of N 3D GMM
# if overlap too much not accurate can fit the data but like the sine example
#the clustering means nothing

#therefore want to seperate the centres

for i in range(N):
    index = i
    centre = 5*np.random.rand(3)
    counter_mean.append(centre)
    
    #cov_mat = make_spd_matrix(n_dim = 3, random_state=None)
    cov_mat = make_sparse_spd_matrix(dim = 3, random_state=None)
    counter_cov_mat.append(cov_mat)
    
    X, Y, Z = np.random.multivariate_normal(mean = centre, cov = cov_mat, size = (number_samples)).T
    counter_x += X
    counter_y += Y
    counter_z += Y
    """#want to also see what the seperated distribution looks like
    #plot the 3D generated samples
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(X, Y, Z)
    threedee.set_xlabel('x' + str(index))
    threedee.set_ylabel('y' + str(index))
    threedee.set_zlabel('z' + str(index))
    plt.show()"""
    

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

fig_1 = threedee.get_figure()
fig_1.suptitle('3D Data', fontsize=16)
fig_1.savefig('Q6.png')
plt.show()


#now retry making a model to predict the GMM

clf_3 = GaussianMixture(n_components=N, covariance_type='full')

#fitting the model to the data
clf_3.fit(gen_df)

#predicting cluster lables

labels_3 = clf_3.predict(gen_df)

#get a handle of the goodness of fit of the model

print("The 3D model had a BIC " + str(clf_3.bic(gen_df)))

print("and the average log likelihood per sample is " + str(clf_3.score(gen_df)))

#apply similar method of adding color labels
threedee = plt.figure().gca(projection='3d')
threedee.scatter(gen_df.x, gen_df.y, gen_df.z, c=labels_3, cmap='viridis')
threedee.set_xlabel('x')
threedee.set_ylabel('y')
threedee.set_zlabel('z')
fig_2 = threedee.get_figure()
fig_2.suptitle('3D Clustered Data', fontsize=16)
fig_2.savefig('Q6_1.png')
plt.show()

#given that we know/can know the averages of these can also work an error from this
#and see if this is a better indicator

#compare the created samples mean and the predicted means 

means_3 = np.array(clf_3.means_)
covariance_3 = clf_3.covariances_

print("The silhouette score is "+ str(silhouette_score(gen_df, labels_3, metric = 'euclidean')))

print("examining the means of the generated GMMs and the predicted ones")
print("counter mean")
print(counter_mean)
print("predicted mean")
print(means_3)
mean_difference = means_3 - counter_mean
print("The differences between the means in this model")
print(mean_difference)
print("This is because the large values dominate the mean")
#Large values are over represented in the mean

#The problem here is that there is far too much overlap in these distributions
#so while the model can fit it the means it provides mean nothing
#want to test that the model would work if the distribution were spread out
