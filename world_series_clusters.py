# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:32:56 2024

@author: dorot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import seaborn as sns
import errors as err
import scipy.optimize as opt


# function to read in data and produce 2 dataframes
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


# a function to create an exponential model for fitting
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 
    and growth rate g.
    """
    # makes it easier to get a guess for initial parameters
    t = t - 1990
    f = n0 * np.exp(g*t)

    return f


# Calling the function with the economic_data CSV file
df1, df2 = read_data('cluster_series.csv')

# replacing empty cells with Nan
df1.replace('..', np.nan, inplace=True)
#print(df1)

# Running the pivot table for 2018 and 2008
df1_2018 = pd.pivot_table(df1, values='2018', index='Country Name',
                              columns='Series Name')
df1_2008 = pd.pivot_table(df1, values='2008', index='Country Name',
                              columns='Series Name')
# dropping nan values in data
df1_2018 = df1_2018.dropna()  
df1_2008 = df1_2008.dropna()

#creating a correlation for the years 2010 and 2018
data2018_corr = df1_2018.corr()
df1_2008_corr = df1_2008.corr()

# plotting the heatmap for 2018
plt.figure()
heatmap = sns.heatmap(data2018_corr, annot=True, cmap='cividis', center=True)
# Increase font size of x-axis labels
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
# Increase font size of y-axis labels and set title
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
heatmap.set_title('Correlation Matrix of Indicators for 2018', fontsize=10, 
              fontweight='bold')

# Creating a scalar object
scaler = pp.RobustScaler()

df18_cluster = df1_2018[['Access to electricity (% of population)', 
                       'Life expectancy at birth, total (years)']]
df10_cluster = df1_2008[['Access to electricity (% of population)', 
                       'Life expectancy at birth, total (years)']]

#normalising the data for 2018 and 2008
normalise18_data = scaler.fit_transform(df18_cluster)
normalise10_data = scaler.fit_transform(df10_cluster)

#plotting graphs of normalised data for 2018 and 2008
plt.figure(figsize = (8,8))
plt.scatter(normalise18_data[:,0], normalise18_data[:, 1], 8, marker='o')

plt.xlabel('Access to electricity')
plt.ylabel('Life Expectancy')

plt.figure(figsize = (8,8))
plt.scatter(normalise10_data[:,0], normalise10_data[:, 1], 8, marker='o')

plt.xlabel('Access to electricity')
plt.ylabel('Life Expectancy')
plt.show()

# a for loop to calculate the silhouette score for clusters
"""
#calculate the silhouette score for 2 to 5 clusters
scores = []
for ic in range(2, 5):
    score = one_silhoutte(normalise_data, ic)
    scores.append([ic, score])
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
"""

#Setting up the cluster with expected clusters
kmeans1 = cluster.KMeans(n_clusters=3, n_init=20, random_state=10)
kmeans2 = cluster.KMeans(n_clusters=3, n_init=20, random_state=10)

#Fit the data, results are stored in the kmeans object
kmeans1.fit(normalise18_data) # fit done on x,y pairs
kmeans2.fit(normalise10_data) 

# extract cluster labels
labels1 = kmeans1.labels_
labels2 = kmeans2.labels_
print(labels1)

#Add cluster labels to the original dataframe
df18_cluster['Cluster'] = labels1
df10_cluster['Cluster'] = labels2

# extract the estimated cluster centres and convert to original scales
cen1 = kmeans1.cluster_centers_
cen1 = scaler.inverse_transform(cen1)
xkmeans = cen1[:, 0]
ykmeans = cen1[:, 1]

# extract the estimated cluster centres and convert to original scales
cen2 = kmeans2.cluster_centers_
cen2 = scaler.inverse_transform(cen2)
xkmeans2 = cen2[:, 0]
ykmeans2 = cen2[:, 1]

# extract x and y values of 2018 data points
x = df18_cluster["Access to electricity (% of population)"]
y = df18_cluster["Life expectancy at birth, total (years)"]
plt.figure()

# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels1, marker="o", cmap = 'Paired')
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d", label = 'kmeans centers')
plt.scatter(xkmeans, ykmeans, 45, "y", marker="+", label = "original centers")

plt.xlabel('Access to Electricity')
plt.ylabel('Life Expectancy at birth')
plt.legend()
plt.title('2018 Cluster graph of Access to Electricity and Life Expectancy',
          fontweight = 'bold', fontsize=10)
plt.show()

# extract x and y values of 2008 data points
x2 = df10_cluster["Access to electricity (% of population)"]
y2 = df10_cluster["Life expectancy at birth, total (years)"]
plt.figure()

# plot data with kmeans cluster number
plt.scatter(x2, y2, 10, labels2, marker="o", cmap='Paired')
# show cluster centres
plt.scatter(xkmeans2, ykmeans, 45, "k", marker="d", label = 'kmeans centers')
plt.scatter(xkmeans2, ykmeans, 45, "y", marker="+", label = "original centers")

plt.xlabel('Access to Electricity')
plt.ylabel('Life Expectancy at birth')
plt.legend()
plt.title('2008 Cluster graph of Access to Electricity and Life Expectancy',
          fontweight = 'bold', fontsize = 10)
plt.show()

# reading in csv file for fitting for countries in cluster 0
df_fit = pd.read_csv('fitting_data.csv')
years = df_fit.Year[0:30]

algeria_data = df_fit[(df_fit['Country Name'] == 'Algeria') & (df_fit['Series Name']
                                == 'Access to electricity (% of population)')]
bangladesh_data = df_fit[(df_fit['Country Name'] == 'Bangladesh') & (df_fit['Series Name']
                                == 'Access to electricity (% of population)')]
china_data = df_fit[(df_fit['Country Name'] == 'China') & (df_fit['Series Name']
                                == 'Access to electricity (% of population)')]

# defining the curve fit for all countries and storing parameters
param1, covar = opt.curve_fit(exponential, algeria_data['Year'], 
                              algeria_data['Access to electricity (% of population)'])
param2, covar = opt.curve_fit(exponential, bangladesh_data['Year'], 
                              bangladesh_data['Access to electricity (% of population)'])
param3, covar = opt.curve_fit(exponential, china_data['Year'], 
                              china_data['Access to electricity (% of population)'])

# plotting the fitting graph for Algeria using the exponential model
param1, covar1 = opt.curve_fit(exponential, algeria_data['Year'], 
                              algeria_data['Access to electricity (% of population)'],
                              p0=(1.2e12, 0.03))
algeria_data["fit"] = exponential(algeria_data["Year"], *param1)
algeria_data.plot("Year", ["Access to electricity (% of population)", "fit"])
plt.title(' Cluster graph of Access to Electricity and Life Expectancy for Algeria',
          fontweight = 'bold', fontsize = 10)
plt.show()

# plotting the fitting graph for Bangladesh using the exponential model
param2, covar2 = opt.curve_fit(exponential, bangladesh_data['Year'], 
                              bangladesh_data['Access to electricity (% of population)'],
                              p0=(1.02e12, 0.03))
bangladesh_data["fit"] = exponential(bangladesh_data["Year"], *param2)
plt.figure()
bangladesh_data.plot("Year", ["Access to electricity (% of population)", "fit"])

plt.title('Fitted Graph of Access to Electricity for Bangladesh',
          fontweight='bold', fontsize=10)
plt.xlabel("Year")
plt.ylabel("Access to electricity")
plt.legend()
plt.show()

# plotting the fitting graph for China using the exponential model
param3, covar3 = opt.curve_fit(exponential, china_data['Year'], 
                              bangladesh_data['Access to electricity (% of population)'],
                              p0=(1.02e12, 0.99938511))
china_data["fit"] = exponential(china_data["Year"], *param3)
china_data.plot("Year", ["Access to electricity (% of population)", "fit"])
plt.show()

# forecasting the data for the next 20 years
forecast_years = np.linspace(1991, 2040, 100)
forecast_values1 = exponential(forecast_years, *param1)
forecast_values2 = exponential(forecast_years, *param2)

#calculating the sigma values using errors function
sigma1 = err.error_prop(forecast_values1, exponential, param1, covar1)
sigma2 = err.error_prop(forecast_values2, exponential, param2, covar2)

#finding limits for confidence intervals
up1 = forecast_values1  
low1 = forecast_values1 

up2 = forecast_values2 + sigma2 
low2 = forecast_values2 - sigma2

#plotting forecast data for Bangladesh
plt.figure()
plt.plot(bangladesh_data["Year"], bangladesh_data["Access to electricity (% of population)"],
         label="Original fit-Access to electricity")
plt.plot(forecast_years, forecast_values2, label="Forecasted fit")
plt.fill_between(forecast_years, low2, up2, color="yellow", alpha=0.7, 
                  label = 'confidence interval')
#setting the labels and titles
plt.xlabel("year")
plt.ylabel("Access to electricity")
plt.title('2040 Forecast of Access to Electricity for Bangladesh',
          fontweight='bold', fontsize=10)
plt.legend()

#plotting forecast data for Algeria
plt.figure(facecolor='lightgrey')
plt.plot(algeria_data["Year"], algeria_data["Access to electricity (% of population)"],
         label="Original fit-Access to electricity")
plt.plot(forecast_years, forecast_values1, label="Forecasted fit")

plt.fill_between(forecast_years, low1, up1, color="yellow", alpha=0.7, 
                  label = 'confidence interval')
#setting the labels and titles
plt.xlabel("year")
plt.ylabel("Access to electricity")
plt.title('2050 Forecast of Access to Electricity for Algeria',
          fontweight='bold', fontsize=10)
plt.legend(facecolor='lightgrey')
plt.show()


