# Preprocessing cluster submodule

## Short description
The module is designed to solve the clustering task. The hdbscan algorithm is used as the core.

____________

## create_clusters function
The algorithm of clustering of points by the coordinates (features in list_columns).

#### Parameters

    'dataframe' (pandas.DataFrame) - dataframe to process
    'min_cluster_size' (int) - minimum amount of objects per cluster
    'min_samples' (int) - the number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    'list_columns' (list) - with names of columns with features to predict cluster labels
    'new_column' (str, default = Cluster) - name of new column in dataframe with cluster labels
    
#### Returns
    
    'dataframe' (pandas Dataframe) - dataframe with cluster labels in new_column
    
____________

## calculate_cluster_centroids
Method for calculating cluster centroids.

#### Parameters

    'dataframe' (pandas.DataFrame) - dataframe to process
    'cluster_column' (str) - name of column in dataframe with cluster labels
    'list_columns' (list) - with names of columns with features to predict cluster labels
    
#### Returns
    
    'dataframe' (pandas Dataframe) - source dataframe with cluster centroids columns. Such columns will have prefix '_centroid', so the full name will be '<COLUMN NAME>_centroid'
    
____________