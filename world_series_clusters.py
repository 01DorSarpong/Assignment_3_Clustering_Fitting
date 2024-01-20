# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:32:56 2024

@author: dorot
"""

import cluster_tools as ct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import seaborn as sns


def read_data(file):
    """
    This function takes in a CSV file as a parameter and returns two
    dataframes, one with columns as years and the other transposed with
    columns as countries.

    Parameters:
        file: This is a CSV file read into the function.
    """

    # Reading the CSV file and setting the index as country name and series name
    df1 = pd.read_csv(file)
    
    # Removing the country code and series code in df1
    df1 = df1.drop(['Country Code', 'Series Code'], axis = 1)
    df1.set_index(['Country Name', 'Series Name'])   
    # Transposing the dataframe 1 and assigning to new dataframe 2
    df2 = df1.T

    # The function returns the two dataframes df1 and df2
    return df1, df2


# a function to get the silhoutte score for clusters
def one_silhoutte(xy, n):
    """
    # This function calculates the silhoutte score for n clusters
    """
    #set up clusters with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20, random_state=10)
    
    #Fit the data and store results in the kmeans object
    kmeans.fit(xy) # fit done on x,y pairs
    
    labels = kmeans.labels_
    score = (skmet.silhouette_score(xy, labels))
    return score


# Calling the function with the economic_data CSV file
df1, df2 = read_data('cluster_series.csv')

# replacing empty cells with Nan
df1.replace('..', np.nan, inplace=True)
#print(df1)

# Running the pivot table for 2018
df1_2010 = pd.pivot_table(df1, values='2010', index='Country Name',
                              columns='Series Name')

# dropping nan values in data
df1_2010 = df1_2010.dropna()  
   
data_corr = df1_2010.corr()
print(data_corr)

plt.figure(figsize=(10, 8))
sns.heatmap(data_corr, annot=True)

# Creating a scalar object
scaler = pp.RobustScaler()

df_cluster = df1_2010[['Access to electricity (% of population)', 
                       'Life expectancy at birth, total (years)']]

normalise_data = scaler.fit_transform(df_cluster)

plt.figure(figsize=(8,8))
plt.scatter(normalise_data[:,0], normalise_data[:, 1], 8, marker='o')

plt.xlabel('Access to electricity')
plt.ylabel('Life Expectancy')

plt.show()

"""
#calculate the silhouette score for 2 to 5 clusters
scores = []
for ic in range(2, 5):
    score = one_silhoutte(normalise_data, ic)
    scores.append([ic, score])
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
"""

#Setting up the cluster with expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20, random_state=10)


#Fit the data, results are stored in the kmeans object
kmeans.fit(normalise_data) # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_

# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

# extract x and y values of data points
x = df_cluster["Access to electricity (% of population)"]
y = df_cluster["Life expectancy at birth, total (years)"]
plt.figure(figsize=(8.0, 6.0))

# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o", cmap='Paired')
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel('Access to electricity')
plt.ylabel('Life Expectancy')
plt.show()



