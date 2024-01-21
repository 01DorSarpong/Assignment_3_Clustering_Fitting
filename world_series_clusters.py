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
df1_2018 = pd.pivot_table(df1, values='2018', index='Country Name',
                              columns='Series Name')

# dropping nan values in data
df1_2018 = df1_2018.dropna()  
   
data_corr = df1_2018.corr()
print(data_corr)

plt.figure(figsize=(10, 8))
sns.heatmap(data_corr, annot=True)

# Creating a scalar object
scaler = pp.RobustScaler()

df_cluster = df1_2018[['Access to electricity (% of population)', 
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
print(labels)


#Add cluster labels to the original dataframe
df_cluster['Cluster'] = labels
#print(df_cluster.head(50))



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
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d", label='kmeans centers')
plt.scatter(xkmeans, ykmeans, 45, "y", marker="+", label="original centers")

plt.xlabel('Access to electricity')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()

"""
# Selecting 3 countries in cluster 0 for fitting
series_name = 'Access to electricity (% of population)'

# Select data for the specified country and series for all years
algeria_data = df1.loc[(df1['Country Name'] == 'Algeria') 
                        & (df1['Series Name'] == series_name), :]
bangladesh_data = df1.loc[(df1['Country Name'] == 'Bangladesh') 
                        & (df1['Series Name'] == series_name), :]
china_data = df1.loc[(df1['Country Name'] == 'China') 
                        & (df1['Series Name'] == series_name), :]
#storing all years in variable years
years = df1.columns[2:]
"""

df_fit = pd.read_csv('fitting_data.csv')
years = df_fit.Year[0:30]



print(df_fit)

algeria_data = df_fit[(df_fit['Country Name'] == 'Algeria') & (df_fit['Series Name']
                                == 'Access to electricity (% of population)')]
bangladesh_data = df_fit[(df_fit['Country Name'] == 'Bangladesh') & (df_fit['Series Name']
                                == 'Access to electricity (% of population)')]
china_data = df_fit[(df_fit['Country Name'] == 'China') & (df_fit['Series Name']
                                == 'Access to electricity (% of population)')]
def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 
    and growth rate g.
    """
    # makes it easier to get a guess for initial parameters
    t = t - 1990
    f = n0 * np.exp(g*t)

    return f

param1, covar = opt.curve_fit(exponential, algeria_data['Year'], 
                              algeria_data['Access to electricity (% of population)'])
param2, covar = opt.curve_fit(exponential, bangladesh_data['Year'], 
                              bangladesh_data['Access to electricity (% of population)'])
param3, covar = opt.curve_fit(exponential, china_data['Year'], 
                              china_data['Access to electricity (% of population)'])
print("GDP 1990", param1[0]/1e9)
print("growth rate", param1[1])

param1, covar1 = opt.curve_fit(exponential, algeria_data['Year'], 
                              algeria_data['Access to electricity (% of population)'],
                              p0=(1.05e12, 0.791))
#(1.2e12, 0.03)

algeria_data["fit"] = exponential(algeria_data["Year"], *param1)
algeria_data.plot("Year", ["Access to electricity (% of population)", "fit"])
plt.show()

# plotting the fitting graph for Bangladesh
param2, covar2 = opt.curve_fit(exponential, bangladesh_data['Year'], 
                              bangladesh_data['Access to electricity (% of population)'],
                              p0=(1.02e12, 0.03))
#(1.2e12, 0.03)

bangladesh_data["fit"] = exponential(bangladesh_data["Year"], *param2)
bangladesh_data.plot("Year", ["Access to electricity (% of population)", "fit"])
plt.show()

# plotting the fitting graph for China
param3, covar3 = opt.curve_fit(exponential, china_data['Year'], 
                              bangladesh_data['Access to electricity (% of population)'],
                              p0=(1.2e12, 00.3))

china_data["fit"] = exponential(china_data["Year"], *param3)
china_data.plot("Year", ["Access to electricity (% of population)", "fit"])
plt.show()

# forecasting the data for the next 20 years
forecast_years = np.linspace(1991, 2050, 100)
forecast_values = exponential(forecast_years, *param2)

# calculating the confidence intervals for new fit
std_dev = np.sqrt(np.diag(covar2))
print(std_dev)

#confidence interval
confidence_int = 0.868

sigma = err.error_prop(forecast_years, exponential, param2, covar2)

up = param2 + confidence_int *std_dev
low = param2 - confidence_int *std_dev

#plt.fill_between(forecast_years, low, up, color="yellow", alpha=0.7)

plt.figure()

plt.fill_between(forecast_years, exponential(forecast_years, low), 
           exponential(forecast_years, up, g='some_val'), color="green", alpha=0.5,
           label = 'confidence interval')

plt.plot(bangladesh_data["Year"], bangladesh_data["Access to electricity (% of population)"],
         label="Original fit-Access to electricity")
plt.plot(forecast_years, forecast_values, label="Forecasted fit")

plt.xlabel("year")
plt.ylabel("Access to electricity")
plt.legend()
plt.show()





"""
#Data Visualisations
top10_GNI=df1_2015['GNI (current US$)'].sort_values(ascending=False)[0:10]
plt.figure(figsize=(15,5))
plt.plot(top10_GNI.index,top10_GNI.values)
plt.title('Top 10 GNI Countries')
plt.show()

top10_labour=df1_2015['Exports of goods and services (current LCU)'].sort_values(ascending=False)[0:10]
plt.figure(figsize=(15,5))
plt.plot(top10_labour.index,top10_labour.values)
plt.title('Top 10 Exports Countries')
plt.show()
"""