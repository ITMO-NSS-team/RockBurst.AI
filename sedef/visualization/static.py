import seaborn as sns
import numpy as np
import pandas as pd

import matplotlib as plt
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_simple(dataframe: pd.DataFrame, columns_to_show: dict,
                   view_init: tuple = (45, 45), cmap: str = 'rainbow') -> None:
    """ The function display simple 3d plot with source data

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    columns_to_show: dict
        columns to plot, where 'x' - for x-axis, 'y' - y-axis, 'z' - z-axis and
        'target' - column name to visualize with color
    view_init: tuple
        view point
    cmap: tuple
        color palette

    Returns
    -------
    plot 3d graph
    """

    fig = plt.figure()
    ax = Axes3D(fig)

    x_vals = np.array(dataframe[columns_to_show.get('x')])
    y_vals = np.array(dataframe[columns_to_show.get('y')])
    z_vals = np.array(dataframe[columns_to_show.get('z')])
    target_arr = np.array(dataframe[columns_to_show.get('target')])

    surf = ax.scatter(x_vals, y_vals, z_vals, c=target_arr, cmap=cmap)
    ax.view_init(view_init[0], view_init[1])
    fig.colorbar(surf, shrink=0.5, aspect=10,
                 label=columns_to_show.get('target'))
    plt.xlabel(columns_to_show.get('x'), fontsize=10)
    plt.ylabel(columns_to_show.get('y'), fontsize=10)
    ax.set_zlabel(columns_to_show.get('z'), fontsize=10)
    plt.show()


def plot_3d_clusters(dataframe: pd.DataFrame, columns_to_show: dict,
                     view_init: tuple = (45, 45), cmap: str = 'tab20c'):
    """ The function display 3d plot with calculated clusters and it's centroids

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    columns_to_show: dict
        columns to plot, where 'x' - for x-axis, 'y' - y-axis, 'z' - z-axis and
        'cluster' - column name to visualize with color
    view_init: tuple
        view point
    cmap: tuple
        color palette

    Returns
    -------
    plot 3d graph
    """
    cluster_name = columns_to_show.get('cluster')
    x_name = columns_to_show.get('x')
    y_name = columns_to_show.get('y')
    z_name = columns_to_show.get('z')

    fig = plt.figure()
    ax = Axes3D(fig)

    x_vals = np.array(dataframe[x_name])
    y_vals = np.array(dataframe[y_name])
    z_vals = np.array(dataframe[z_name])
    cluster_arr = np.array(dataframe[cluster_name])

    # For every cluster we also need to plot centroid
    x_coords = []
    y_coords = []
    z_coords = []
    labels = []
    for label in dataframe[cluster_name].unique():
        cluster_df = dataframe[dataframe[cluster_name] == label]
        x_centroid_name = ''.join((x_name, '_centroid'))
        y_centroid_name = ''.join((y_name, '_centroid'))
        z_centroid_name = ''.join((z_name, '_centroid'))

        local_x_arr = np.array(cluster_df[x_centroid_name])
        local_y_arr = np.array(cluster_df[y_centroid_name])
        local_z_arr = np.array(cluster_df[z_centroid_name])

        x_coords.append(local_x_arr[0])
        y_coords.append(local_y_arr[0])
        z_coords.append(local_z_arr[0])
        labels.append(label)

    surf = ax.scatter(x_vals, y_vals, z_vals, c=cluster_arr, cmap=cmap)
    centr = ax.scatter(x_coords, y_coords, z_coords, c=labels, cmap=cmap, s=100,
                       marker='D')
    ax.view_init(view_init[0], view_init[1])
    fig.colorbar(surf, shrink=0.5, aspect=10,
                 label=cluster_name)
    plt.xlabel(columns_to_show.get('x'), fontsize=10)
    plt.ylabel(columns_to_show.get('y'), fontsize=10)
    ax.set_zlabel(columns_to_show.get('z'), fontsize=10)
    plt.show()
