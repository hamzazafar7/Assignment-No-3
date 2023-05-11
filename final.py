#Assignment No 03
#Muhammad Hamza Zafar
#Student ID: 22022247
#Applied Data Science


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.optimize import curve_fit
import itertools as iter

def read_data(filename: str):
    
    
    """
    Reads a CSV file into a pandas dataframe and performs data transformation.

    Args:
        filename (str): The name of the CSV file to be read.

    Returns:
        df (pd.DataFrame): The original dataframe read from the CSV file.
        df_country (pd.DataFrame): The transposed dataframe with countries as columns.
        df_year (pd.DataFrame): The transposed dataframe with years as columns.
    """
    
    # Read the file into a pandas dataframe
    df = pd.read_csv(filename)
    
    # Transpose the dataframe
    df_trans = df.transpose()
    
    # Populate the header of the transposed dataframe with the header information 
    # slice the dataframe to get the year as columns
    df_trans.columns = df_trans.iloc[1]

    # As year is now columns so we don't need it as rows
    df_trans_year = df_trans[0:].drop('year')
    
    # Slice the dataframe to get the country as columns
    df_trans.columns = df_trans.iloc[0]
    
    # As country is now columns so we don't need it as rows
    df_trans_country = df_trans[0:].drop('country')
    
    return df, df_trans_country, df_trans_year

# load data from World Bank website or a similar source
df, df_country, df_year = read_data('worldbank_climatechange.csv')


#  removes null values from a given feature.
def remove_null_values(feature):
    
    
    """
    Removes null values from a pandas Series and returns a NumPy array.

    Args:
        feature (pd.Series): The pandas Series containing the feature.

    Returns:
        np.ndarray: NumPy array containing the non-null values from the feature.
    """
    
    # Drop null values from the feature
    return np.array(feature.dropna())

def balance_data(df: pd.DataFrame):
    

    """
    Balances the data by removing null values from specific features and 
    creating a new dataframe.

    Args:
        df (pd.DataFrame): The original dataframe containing the features.

    Returns:
        pd.DataFrame: A new dataframe with balanced data, containing non-null 
        values from the specified features.
    """
    
    
    # Making dataframe of all the features available in the dataframe
    # Passing it to the remove_null_values function for dropping the null values
    arable_land = remove_null_values(df[['arable_land']])
    population_growth = remove_null_values(df[['population_growth']])
    forest_area = remove_null_values(df[['forest_area']])

    # Determine the minimum length among the features after removing null values
    min_length = min(len(arable_land), len(population_growth), len(forest_area))

    # Create a new dataframe with balanced data
    clean_data = pd.DataFrame({
        'country': [df['country'].iloc[x] for x in range(min_length)],
        'year': [df['year'].iloc[x] for x in range(min_length)],
        'arable_land': [arable_land[x][0] for x in range(min_length)],
        'forest_area': [forest_area[x][0] for x in range(min_length)],
        'population_growth': [population_growth[x][0] for x in range(min_length)]
    })

    return clean_data

# Clean and preprocess the data
df, _, _ = read_data('worldbank_climatechange.csv')
df = balance_data(df)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
transform_data = scaler.fit_transform(df[['arable_land', 'forest_area', 
                                          'population_growth']])

# Use KMeans to find clusters in the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(transform_data)

# Add the cluster assignments as a new column to the dataframe
df['cluster'] = kmeans.labels_

# Define a function for creating scatter plots with clusters and cluster centers
def plot_clusters(x, y, xlabel, ylabel, title):
    
    """
    Creates a scatter plot showing clusters and cluster centers.

    Args:
        x (str): Name of the column to be used for the x-axis data.
        y (str): Name of the column to be used for the y-axis data.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    
    # Create a scatter plot for each cluster
    for i in range(3):
        cluster_data = df[df['cluster'] == i]
        plt.scatter(cluster_data[x], cluster_data[y], label=f'Cluster {i}')

    # Plot the cluster centers
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                marker='x', c='black', label='Cluster Centers')

    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()

# Create scatter plot for arable_land vs forest_area
plot_clusters('arable_land', 'forest_area', 'Arable Land', 'Forest Area', 'Clusters: Arable Land vs Forest Area')

# Create scatter plot for forest_area vs population_growth
plot_clusters('forest_area', 'population_growth', 'Forest Area', 'Population Growth', 'Clusters: Forest Area vs Population Growth')

ar_land_data = df[['country', 'year','arable_land']]

# data related to 1995
arland_data_1995 = ar_land_data[ar_land_data['year'] == 1995] 

# data related to 2000
arland_data_2000 = ar_land_data[ar_land_data['year'] == 2000] 

# data related to 2005 
arland_data_2005 = ar_land_data[ar_land_data['year'] == 2005] 

# data related to 2010 
arland_data_2010 = ar_land_data[ar_land_data['year'] == 2010]

# data related to 2015 
arland_data_2015 = ar_land_data[ar_land_data['year'] == 2015]

# data related to 2020 
arland_data_2020 = ar_land_data[ar_land_data['year'] == 2020]


# ### Plot Barplot

style.use('ggplot')

# set fig size
plt.figure(figsize=(10,5))

# set width of bars
barWidth = 0.1

# plot bar charts
plt.bar(np.arange(arland_data_1995.shape[0])+0.2,
        arland_data_1995['arable_land'],
        color='yellow',width=barWidth, label='1995')

plt.bar(np.arange(arland_data_2000.shape[0])+0.3,
        arland_data_2000['arable_land'],
        color='red',width=barWidth, label='2000')

plt.bar(np.arange(arland_data_2005.shape[0])+0.4,
        arland_data_2005['arable_land'],
        color='blue',width=barWidth, label='2005')

plt.bar(np.arange(arland_data_2010.shape[0])+0.5,
        arland_data_2010['arable_land'],
        color='indigo',width=barWidth, label='2010')

plt.bar(np.arange(arland_data_2015.shape[0])+0.6,
        arland_data_2015['arable_land'],
        color='darkslategray',width=barWidth, label='2015')

plt.bar(np.arange(arland_data_2020.shape[0])+0.7,
        arland_data_2020['arable_land'],
        color='darkviolet',width=barWidth, label='2020')



# show the legends on the plot
plt.legend()

# set the x-axis label
plt.xlabel('Country',fontsize=15)

# add title to the plot 
plt.title("Arable Land",fontsize=15)

# add countries names to the 11 groups on the x-axis
plt.xticks(np.arange(arland_data_2015.shape[0])+0.2,
           ('Australia', 'China', 'Germany', 'India', 'Japan', 'Mexico',
       'Netherlands', 'Russian Federation', 'Turkiye', 'Ukraine',
       'United States', 'South Africa'),
           fontsize=10,rotation = 45)

# show the plot
plt.show()



ind = df[df['country'] == 'India']

# create a plot showing the clusters and cluster centers using pyplot
for i in range(3):
    cluster_data = ind[ind['cluster'] == i]
    plt.scatter(cluster_data['forest_area'], cluster_data['population_growth'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Forest Area')
plt.ylabel('Population Growth')
plt.title('Clusters')
plt.legend()
plt.show()

sa = df[df['country'] == 'South Africa']

# create a plot showing the clusters and cluster centers using pyplot
for i in range(3):
    cluster_data = sa[sa['cluster'] == i]
    plt.scatter(cluster_data['arable_land'], cluster_data['forest_area'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', c='black', label='Cluster Centers')
plt.xlabel('Arable Land')
plt.ylabel('Forest Area')
plt.title('Cluster')
plt.legend()
plt.show()

def err_ranges(x, func, param, sigma):
    
    
    """
    Computes the lower and upper limits of the error range for a given function and parameter uncertainties.

    Args:
        x (np.ndarray): The input array for the function.
        func (function): The function to compute the error range for.
        param (tuple): The parameter values for the function.
        sigma (tuple): The uncertainties or standard deviations for each parameter.

    Returns:
        np.ndarray: The lower limit of the error range for the function.
        np.ndarray: The upper limit of the error range for the function.
    """
    # Initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []  # List to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

# Define the exponential function
def exp_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    
    
    """
    Evaluates the exponential function for the given input array.

    Args:
        x (np.ndarray): Input array for the function.
        a (float): Parameter a of the exponential function.
        b (float): Parameter b of the exponential function.

    Returns:
        np.ndarray: Output array computed by the exponential function.
    """
    return a * np.exp(b * x)


# Data of cluster 2
c1 = df[(df['cluster'] == 2)]

# x values and y values
x = c1['arable_land']
y = c1['forest_area']

popt, pcov = curve_fit(exp_func, x, y)

# Use err_ranges function to estimate lower and upper limits of the confidence range
sigma = np.sqrt(np.diag(pcov))
lower, upper = err_ranges(x, exp_func, popt,sigma)

# Use pyplot to create a plot showing the best fitting function and the confidence range
plt.plot(x, y, 'o', label='data')
plt.plot(x, exp_func(x, *popt), '-', label='fit')
plt.fill_between(x, lower, upper, color='pink', label='confidence interval')
plt.legend()
plt.xlabel('Arable Land')
plt.ylabel('Forest Area')
plt.title('Confidence Interval')
plt.show()

# Define the range of future x-values for which you want to make predictions
future_x = np.arange(30, 40)

# Use the fitted function and the estimated parameter values to predict the future y-values
future_y = exp_func(future_x, *popt)

# Plot the predictions along with the original data
plt.plot(x, y, 'o', label='data')
plt.plot(x, exp_func(x, *popt), '-', label='fit')
plt.plot(future_x, future_y, 'o', label='future predictions')
plt.xlabel('Arable Land')
plt.title('Future predictions')
plt.ylabel('Forest Area')
plt.legend()
plt.show()