import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def interactive_3d_simple(dataframe: pd.DataFrame, columns_to_show: dict) -> None:
    """ The function display simple 3d plot with source data

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    columns_to_show: dict
        columns to plot, where 'x' - for x-axis, 'y' - y-axis, 'z' - z-axis and
        'target' - column name to visualize with color

    Returns
    -------
    plot 3d graph with plotly
    """

    fig = px.scatter_3d(dataframe, x=columns_to_show.get('x'),
                        y=columns_to_show.get('y'),
                        z=columns_to_show.get('z'),
                        color=columns_to_show.get('target'))
    fig.show()


def interactive_line_plot(dataframe: pd.DataFrame, columns_to_show: dict) -> None:
    """ The function display simple 3d plot with source data

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    columns_to_show: dict
        columns to plot, where 'x' - for x-axis, 'y' - y-axis, 'z' - z-axis and
        'target' - column name to visualize with color

    Returns
    -------
    plot 3d graph with plotly
    """

    df = dataframe.dropna()

    fig = px.line(df, x=columns_to_show.get('x'), y=columns_to_show.get('y'),
                  color=columns_to_show.get('color'))
    fig.show()
