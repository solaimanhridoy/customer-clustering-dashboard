# Clustering Dashboard Documentation


## Overview
This document provides a brief overview of the clustering dashboard project. It explains the dataset used, the clustering algorithms implemented, and how the interactive dashboard works.

## Important Links
Please check the deployed dashboard in this [dashboard link](https://web-production-b121.up.railway.app/) and wait for at least 10 seconds to reload the dashboard properly. Thanks!

The Source code is also available in this github [repository link](https://github.com/solaimanhridoy/customer-clustering-dashboard/)

## Dataset Used
The dataset used in this project is the **Mall Customers** dataset. Key details include:

- **Source:**  
  The dataset is available on Kaggle (e.g., [Mall Customers Dataset](https://www.kaggle.com/datasets/shwetabh123/mall-customers)).  
- **Description:**  
  This dataset contains customer information from a mall, which is commonly used for customer segmentation analysis.
- **Key Features:**  
  - **CustomerID:** Unique identifier for each customer.
  - **Genre:** Gender of the customer (Male or Female).
  - **Age:** Age of the customer.
  - **Annual Income (k$):** Annual income of the customer in thousands of dollars.
  - **Spending Score (1-100):** A score (between 1 and 100) assigned based on the customer’s spending behavior.

The dataset is ideal for clustering because it provides a mix of demographic and spending information that can reveal natural groupings among customers.

## Clustering Algorithms Implemented
The dashboard supports multiple clustering algorithms to offer different perspectives on the data:

- **K-Means Clustering:**
  - **Description:**  
    K-Means is a centroid-based algorithm that partitions the dataset into a specified number of clusters by minimizing the sum of squared distances (inertia) between data points and their respective cluster centroids.
  - **Dashboard Usage:**  
    Users can adjust the number of clusters using a slider. An Elbow Method plot is provided to help determine the optimal number of clusters based on inertia values.

- **Hierarchical Clustering:**
  - **Description:**  
    Hierarchical clustering builds a tree-like structure (dendrogram) that represents nested clusters. Although it can be computationally intensive, it is useful for visualizing the data’s natural groupings.
  - **Dashboard Usage:**  
    The dashboard allows users to choose the number of clusters to extract from the dendrogram, facilitating an exploratory analysis of the data’s structure.

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**
  - **Description:**  
    DBSCAN groups data points based on density, identifying clusters of arbitrary shapes and marking points in low-density regions as noise.
  - **Dashboard Usage:**  
    Users can input values for `eps` (the radius for neighborhood search) and `min_samples` (the minimum number of points required to form a dense region), enabling fine-tuning of the clustering process.

## How the Dashboard Works
The dashboard is built using **Dash** and **Dash Bootstrap Components** in Python. Key functionalities include:

- **Dynamic Filters:**
  - **Gender Filter:**  
    A dropdown lets users filter the dataset based on gender.
  - **Age Range Slider:**  
    A slider enables users to select a specific age range, focusing the analysis on a particular demographic.

- **Clustering Options:**
  - **Algorithm Selection:**  
    A dropdown menu allows users to choose among K-Means, Hierarchical Clustering, and DBSCAN.
  - **Parameter Inputs:**  
    Depending on the selected algorithm, the dashboard dynamically displays:
    - A slider for the number of clusters (for K-Means and Hierarchical Clustering).
    - Input fields for `eps` and `min_samples` (for DBSCAN).

- **Interactive Visualizations:**
  - **Scatter Plot:**  
    The dashboard displays a scatter plot of Annual Income versus Spending Score. Each customer is color-coded according to the assigned cluster.
  - **Elbow Method Plot:**  
    For K-Means clustering, an additional plot shows inertia values for different numbers of clusters, aiding in selecting an optimal cluster count.

- **Deployment:**  
  The app is configured to run on `0.0.0.0` and uses a dynamic port (via an environment variable), making it suitable for deployment on free hosting platforms like Railway (railway.app).

## Conclusion
This clustering dashboard is designed as an interactive tool for customer segmentation analysis. By integrating multiple clustering algorithms with dynamic filtering and visualization features, it allows users to explore and gain insights into customer behavior. This project provides a solid foundation for further exploration and extension in real-world data analysis scenarios.

---
