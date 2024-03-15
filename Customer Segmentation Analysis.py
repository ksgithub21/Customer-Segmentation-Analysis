#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation Analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[3]:


customer = pd.read_csv(r"C:\Users\hp\Documents\Komal Singh Files\archive (1)\Mall_Customers.csv")


# In[4]:


customer.head()


# In[5]:


customer.isnull().sum()


# In[6]:


customer.shape


# In[7]:


customer.info()


# In[8]:


customer.isnull().sum()


# In[9]:


X = customer.iloc[:,[3,4]].values
print(X)


# Cluster , WCSS(Within Cluster Sum Of Square)

# In[13]:


wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(X)

  wcss.append(kmeans.inertia_)


# In[15]:


sb.set()
plt.plot(range(1,11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# Optimum Number of Clusters = 5
# 
# Training the k-Means Clustering Model

# In[16]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

print(Y)


# Visualizing all the Clusters

# In[19]:


# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0], X[Y==0,1], s=50, c='purple', label='Cluster 1')
plt.scatter(X[Y==1,0], X[Y==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y==2,0], X[Y==2,1], s=50, c='black', label='Cluster 3')
plt.scatter(X[Y==3,0], X[Y==3,1], s=50, c='violet', label='Cluster 4')
plt.scatter(X[Y==4,0], X[Y==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='cyan', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()

