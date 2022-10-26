import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_clusters(dataframe, min_cluster_size: int, min_samples: int,
                    list_columns: list = ['X', 'Y', 'Z'],
                    new_column: str = 'Cluster'):
    """
    The algorithm of clustering of points by the coordinates -

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    min_cluster_size : int
        minimum amount of objects per cluster
    min_samples : int
        The number of samples (or total weight) in a neighborhood for a point to
        be considered as a core point.
        How conservative you want you clustering to be. The larger the value of
        min_samples you provide, the more conservative the clustering – more
        points will be declared as noise, and clusters will be restricted to
        progressively more dense areas.
    list_columns : list
        array-like with names of columns with features to predict cluster labels
    new_column : str
        name of new column in dataframe with cluster labels

    Returns
    -------
    dataframe : pd.DataFrame
        source dataframe with cluster labels
    """

    x = np.array(dataframe[list_columns])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                 min_samples=min_samples).fit(x_scaled)
    dataframe[new_column] = clustering.labels_
    return dataframe


def calculate_cluster_centroids(dataframe: pd.DataFrame, cluster_column: str,
                                list_columns: list):
    """
    Method for calculating cluster centroids

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    cluster_column : str
        name of column in dataframe with cluster labels
    list_columns : list
        array-like with names of columns with features to predict cluster labels

    Returns
    -------
    dataframe : pd.DataFrame
        source dataframe with cluster centroids columns. Such columns will have
        prefix '_centroid', so the full name will be '<COLUMN NAME>_centroid'
    """

    # Iterating over cluster labels
    labels = dataframe[cluster_column].unique()

    # Calculating cluster centroids for every feature
    centroid_feature = {}
    for label in labels:
        # Local part of dataframe with only one cluster
        cluster_data = dataframe[dataframe[cluster_column] == label]

        # We take the minimum and maximum values for each coordinate (feature)
        coord_centroids = {}
        for feature in list_columns:
            # Centroids per cluster
            centroid = (max(cluster_data[feature])+min(cluster_data[feature]))/2
            coord_centroids.update({feature: centroid})
        centroid_feature.update({label: coord_centroids})

    # Assigning values part
    centroid_names = []
    for feature in list_columns:
        centroid_name = ''.join((feature, '_centroid'))
        feature_centroid_column = []

        # For every cluster assign appropriate centroid value
        for current_cluster in dataframe[cluster_column]:

            cluster_info = centroid_feature.get(current_cluster)
            feature_centroid_data = cluster_info.get(feature)

            feature_centroid_column.append(feature_centroid_data)

        dataframe[centroid_name] = feature_centroid_column
        centroid_names.append(centroid_name)

    return dataframe
