import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from scipy import optimize
from copy import copy

from matplotlib import pyplot as plt
import datetime


class DiscreteStepSearch:
    """
    Class for finding the optimal time step for the selected dataset

    Parameters
    ----------
    dataframe : pandas DataFrame
        the dataframe for processing
    collision_w : float
        weight of the collision case when averaging (collision_w + gaps_w = 1)
    gaps_w : float
        weight of the omission case when averaging (collision_w + gaps_w = 1)
    vis : bool
        is it necessary to visualize the solutions
    """

    def __init__(self, dataframe, collision_w=0.5, gaps_w=0.5, vis=False):
        self.df = dataframe
        self.collision_w = collision_w
        self.gaps_w = gaps_w

        # Is it necessary to visualize the solutions
        self.vis = vis

        self.time_steps_arr = []
        self.collisions_arr = []
        self.gaps_arr = []
        self.scores = []

        # Attributes, which used during the class processing
        self.seconds_arr = None
        self.start_seconds = None
        self.final_seconds = None

    def time_step_search(self, datetime_column='Datetime',
                         method='golden') -> float:
        """ A method that allows determining the optimal time step for sampling

        Parameters
        ----------
        datetime_column : str
            the name of the column that is present in the dataset and contains
            the datetime
        method : str
            optimal solution search method ('golden' or 'brute')
        Returns
        -------
        datetime as float
            Suggested optimal time step in seconds
        """

        # The starting date that is in the dataframe
        start_datetime = self.df[datetime_column].min()

        # Array of starting points
        len_data = len(self.df)
        self.df['Start_points'] = [start_datetime] * len_data
        self.df['Start_points'] = pd.to_datetime(self.df['Start_points'])

        # Calculation of the time in seconds
        abs_time_start = self.df[datetime_column] - self.df['Start_points']
        abs_time = []
        for i in abs_time_start:
            # Total time in seconds
            sec_time = i.total_seconds()
            abs_time.append(sec_time)
        self.seconds_arr = np.array(abs_time)
        self.df.drop(columns='Start_points', inplace=True)

        # The definition of the boundaries of interval
        self.start_seconds = np.min(self.seconds_arr)
        self.final_seconds = np.max(self.seconds_arr)

        five_percent = int(len(self.seconds_arr) * 0.05)
        if method == 'brute':
            # Running the algorithm - brute force search
            all_intervals = range(2, len(self.seconds_arr) + five_percent)
            for i in all_intervals:
                self._eval_score(i)
        elif method == 'golden':
            right_border = len(self.seconds_arr) + five_percent
            # Running the algorithm - golden method
            optimize.minimize_scalar(self._eval_score,
                                     bracket=[2, right_border],
                                     method='golden')

        # Best solution
        self.scores = np.array(self.scores)
        best_id = int(np.argmin(self.scores))

        opt_step = self.time_steps_arr[best_id]
        opt_gap_score = self.gaps_arr[best_id]
        opt_col_score = self.collisions_arr[best_id]
        opt_com_score = self.scores[best_id]

        print(f'For a time step {opt_step:.0f} sec. gap score '
              f'(amount of gaps) - {opt_gap_score:.0f} and collision score '
              f'(amount of elements in collisions/amount of collisions) - {opt_col_score:.1f}'
              f'\nOptimal common score - {opt_com_score:.1f}')

        if self.vis:
            plt.scatter(self.time_steps_arr, self.gaps_arr,
                        c='blue', alpha=0.7, label='Gaps score')
            plt.scatter(self.time_steps_arr, self.collisions_arr,
                        c='green', alpha=0.7, label='Collision score')
            plt.scatter(self.time_steps_arr, self.scores, s=90,
                        c='red', alpha=0.9, label='Common score')
            plt.legend(fontsize=15)
            plt.xlabel('Time step, sec', fontsize=15)
            plt.ylabel('Score, points', fontsize=15)
            plt.grid()
            plt.show()

        return opt_step

    def _eval_score(self, intervals) -> None:
        """A method that allows counting the number of collisions and
        omissions for a given time step and translate it into the fitness
        metric of the selected solution

        Parameters
        ----------
        intervals : float
            the number of intervals to divide the array into
        """

        # The number of intervals is always an integer
        intervals = int(round(intervals))

        # All values of the number of intervals less than 1 are incorrect
        if intervals > 1:
            self.__calculations_valid_solution(intervals)
        else:
            pass

    def __calculations_valid_solution(self, intervals):
        # Generating a massive with the suggested number of iterations
        start = self.start_seconds
        end = self.final_seconds

        proposed_split, estim_timestep = np.linspace(start, end, intervals,
                                                     retstep=True)

        # Determine the number of collisions and omissions in the
        # proposed partition
        gaps = []
        collisions = []
        for index in range(0, len(proposed_split)):
            if index == 0:
                pass
            else:
                left_border = proposed_split[index - 1]
                right_border = proposed_split[index]

                # If the element is the last one, then it is counted
                # inclusive "on the right" - like (...; ...]
                if index == len(proposed_split) - 1:
                    ids_bool_left = np.ravel(
                        np.argwhere(self.seconds_arr >= left_border))
                    ids_bool_right = np.ravel(
                        np.argwhere(self.seconds_arr <= right_border))
                else:
                    ids_bool_left = np.ravel(
                        np.argwhere(self.seconds_arr >= left_border))
                    ids_bool_right = np.ravel(
                        np.argwhere(self.seconds_arr < right_border))

                # Indexes of matching elements
                ids_intersections = np.intersect1d(ids_bool_left,
                                                   ids_bool_right)

                if len(ids_intersections) == 0:
                    # Gap
                    gaps.append(1)
                elif len(ids_intersections) == 1:
                    # There is one element per time interval
                    pass
                else:
                    # Collision
                    collisions.append(len(ids_intersections))

        collisions = np.array(collisions)

        # Total number of passes
        score_gap = len(gaps)

        # How many elements were in the collisions / the number of intervals
        score_collisions = collisions.sum() / len(collisions)

        score = (score_gap * self.gaps_w)+(score_collisions * self.collision_w)

        # Updating statistics
        self.time_steps_arr.append(estim_timestep)
        self.collisions_arr.append(score_collisions)
        self.gaps_arr.append(score_gap)
        self.scores.append(score)


def discretize_time_grid(dataframe: pd.DataFrame,
                         start_date: str,
                         end_date: str,
                         old_column: str = 'Datetime',
                         new_column: str = 'New_datetime',
                         time_step: str = '1H'):
    """ The function generates a new column with datetime values with specified
    frequency

    Parameters
    ----------
    dataframe : pandas DataFrame
        dataframe to process
    start_date: str
        date time of the start of the time interval for sampling in the format
        "%Y-%m-%d %H:%M:%S"
    end_date: str
        date time of the end of the time interval for sampling in the format
        "%Y-%m-%d %H:%M:%S"
    old_column : str
        the name of the column that is present in the dataset and contains the
        date/time
    new_column : str
        the column that you want to generate
    time_step: str
        the time step for which you want to cause a time series. For example,
        '5min' - 5 minutes, '5H' - 5 hours, '5D' - 5 days, etc.

    Returns
    -------
    pandas DataFrame
        Dataframe with an additional new_column column that contains equidistant
        time series labels
    """

    dataframe.sort_values(by=[old_column], inplace=True)
    shape = dataframe.shape

    print(f'\nThe time series will be composed with frequency - {time_step}')
    print(f'Start date - {start_date}')
    print(f'Final date - {end_date}')

    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    # Generating a time series with equal pre-defined intervals
    times = pd.date_range(start=start_date, end=end_date, freq=time_step)

    # The trigger responds to whether the algorithm has reached the first valid
    # time interval in the dataset
    trigger = 0
    for i in range(0, len(times) - 1):
        # Taking the centroid from the selected period
        time_interval = (times[i + 1] - times[i]) / 2
        centroid = times[i] + time_interval

        # Which rows are suitable for this time interval
        local_df = dataframe[dataframe[old_column] >= times[i]]
        local_df = local_df[local_df[old_column] < times[i + 1]]

        # If no point is suitable for a given timestamp, then
        if len(local_df) == 0:
            # If the timestamp exceeds the maximum date/time stamp in the
            # dataframe, then we end the loop
            if times[i] > max(dataframe[old_column]):
                break
            # If the timestamp is less than the minimum timestamp in the
            # dataframe, then the algorithm just hasn't reached the "right"
            # moment yet
            elif times[i+1] <= min(dataframe[old_column]):
                continue
            else:
                empty_row = np.full((1, shape[1]), np.nan)
                local_df = pd.DataFrame(empty_row, columns=dataframe.columns)
                local_df[new_column] = [centroid] * len(local_df)

                # Adding an empty local dataframe
                frames = [new_dataframe, local_df]
                new_dataframe = pd.concat(frames)

        # If there are at least some suitable objects
        else:
            local_df[new_column] = [centroid] * len(local_df)

            # Declaring a dataframe on the first iteration
            if trigger == 0:
                new_dataframe = local_df
                trigger = 1
            else:
                # Then, attach the other parts to it
                frames = [new_dataframe, local_df]
                new_dataframe = pd.concat(frames)

    return new_dataframe


def merge_and_tens(dataframe: pd.DataFrame,
                   cluster_column: str,
                   new_time_column: str,
                   old_time_column: str,
                   list_columns: list,
                   target_regression: str = None,
                   target_classification: str = None):
    """ The function "pulls" objects in the dataframe to the specified regular
    labels in space and time

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    cluster_column : str
        name of the column that contains the cluster labels
    new_time_column : str
        name of column with new datetime steps
    old_time_column: str
        name of column with old datetime steps
    list_columns : list
        array-like with names of columns with features to predict cluster labels
    target_regression: str
        column name with float target values
    target_classification: str
        column name with categorical target values

    Returns
    -------
    pandas DataFrame
        Dataframe with sampled values in time and space
    """

    # Define what type of task it will be
    if target_regression is not None and target_classification is not None:
        task_type = 'both'
    elif target_regression is not None and target_classification is None:
        task_type = 'regression'
    elif target_regression is None and target_classification is not None:
        task_type = 'classification'
    else:
        raise ValueError('At least one target must be defined')

    # Iteratively, shift the centroids across the space by time equal-spaced
    # time indices
    dataframe.sort_values(by=[new_time_column], inplace=True)
    original_columns = list(dataframe.columns)
    original_columns.append('distance')

    # Features for K-nn models
    updated_list_columns = copy(list_columns)
    updated_list_columns.append('distance')

    # Names of columns with centroids coordinates
    centroid_names = []
    for feature in list_columns:
        centroid_name = ''.join((feature, '_centroid'))
        centroid_names.append(centroid_name)

    for i, time_index in enumerate(dataframe[new_time_column].unique()):
        local_df = dataframe[dataframe[new_time_column] == time_index]

        # Convert time to absolute values (seconds)
        local_df['distance'] = _time_deviation(dataframe=local_df,
                                               new_time_column=new_time_column,
                                               old_time_column=old_time_column)

        for j, label in enumerate(local_df[cluster_column].unique()):
            # Dataframe for a specific cluster at a specific time index
            local_cluster_data = local_df[local_df[cluster_column] == label]

            # Current information for iteration
            current_info = {'label': label,
                            'time_index': time_index,
                            'original_columns': original_columns,
                            'cluster_column': cluster_column,
                            'new_time_column': new_time_column,
                            'task_type': task_type}

            # Make merge and tens operation
            new_object = _make_pull(local_cluster_df=local_cluster_data,
                                    list_columns=updated_list_columns,
                                    centroid_names=centroid_names,
                                    target_regression=target_regression,
                                    target_classification=target_classification,
                                    current_info=current_info)

            if i == 0 and j == 0:
                # Create dataframe
                new_dataframe = new_object
            else:
                frames = [new_dataframe, new_object]
                new_dataframe = pd.concat(frames)

    new_dataframe.drop(columns=['distance'], inplace=True)
    return new_dataframe


def _time_deviation(dataframe, new_time_column, old_time_column):
    """
    The function calculates the difference between the old and new timestamp
    values in absolute values (amount of seconds)

    Parameters
    ----------
    dataframe : pd.DataFrame
        dataframe to process
    new_time_column : str
        name of column with new datetime steps
    old_time_column : str
        name of column with old datetime steps

    Returns
    -------
    diff : array
        array with deviations of the old timestamps from the new one in seconds
    """

    # Converting the datetime to absolute values (in seconds)
    new_times = pd.to_datetime(dataframe[new_time_column])
    old_times = pd.to_datetime(dataframe[old_time_column])
    # There are 24 hours per day, 60 minutes per hour, and
    # 60 seconds per minute
    new_part = (new_times.dt.day * 24 + new_times.dt.hour) * 60 + new_times.dt.minute
    new_times = new_part * 60 + new_times.dt.second
    old_part = (old_times.dt.day * 24 + old_times.dt.hour) * 60 + old_times.dt.minute
    old_times = old_part * 60 + old_times.dt.second

    # Subtract from the time series with "new" labels - "old"
    diff = abs(np.array(new_times) - np.array(old_times))

    return diff


def _make_pull(local_cluster_df: pd.DataFrame, list_columns: list,
               centroid_names: list, target_regression: str,
               target_classification: str, current_info: dict):
    """

    Parameters
    ----------
    local_cluster_df : pd.DataFrame
        dataframe for specific cluster and time index
    list_columns : list
        array-like with names of columns with features to predict cluster labels
    target_regression: str
        column name with float target values
    target_classification: str
        column name with categorical target values
    current_info :
        dictionary with information about current iteration

    Returns
    -------
        pandas dataframe as one row
    """

    original_columns = current_info.get('original_columns')
    current_label = current_info.get('label')
    current_time_id = current_info.get('time_index')
    task_type = current_info.get('task_type')

    # Length of the dataframe - how many objects in collision
    local_len = len(local_cluster_df)

    if local_len == 0:
        # Create empty row
        empty_row = np.full((1, len(original_columns)), np.nan)
        new_object = pd.DataFrame(empty_row, columns=original_columns)

        # Include several values in this empty dataframe
        new_object[current_info.get('cluster_column')] = current_label
        new_object[current_info.get('new_time_column')] = current_time_id
    elif local_len == 1:
        # If there is only one object for the cluster - shift it to the centroid
        new_object = local_cluster_df.copy()
    else:
        new_object = local_cluster_df.copy()
        # Use K-nn for predicting target values (float or category) in centroid
        x_train = np.array(local_cluster_df[list_columns])
        x_test = np.array(local_cluster_df[centroid_names])

        # Append zeros column, cause there is no difference in time index
        zeros_col = np.zeros((local_len, 1))
        x_test = np.hstack((x_test, zeros_col))

        # Standardizing the samples
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        if task_type == 'both':
            # Regression
            y_train_float = np.array(local_cluster_df[target_regression])
            neighbors_regressor = KNeighborsRegressor(n_neighbors=local_len)
            neighbors_regressor.fit(x_train, y_train_float)
            predicted_float = neighbors_regressor.predict(x_test)
            new_object[target_regression] = predicted_float

            # Classification
            y_train_class = np.array(local_cluster_df[target_classification])
            neighbors_classifier = KNeighborsClassifier(n_neighbors=local_len)
            neighbors_classifier.fit(x_train, y_train_class)
            predicted_class = neighbors_classifier.predict(x_test)
            new_object[target_classification] = predicted_class

        elif task_type == 'regression':
            y_train_float = np.array(local_cluster_df[target_regression])
            neighbors_regressor = KNeighborsRegressor(n_neighbors=local_len)

            neighbors_regressor.fit(x_train, y_train_float)
            predicted_float = neighbors_regressor.predict(x_test)

            new_object[target_regression] = predicted_float

        elif task_type == 'classification':
            y_train_class = np.array(local_cluster_df[target_classification])
            neighbors_classifier = KNeighborsClassifier(n_neighbors=local_len)

            neighbors_classifier.fit(x_train, y_train_class)
            predicted_class = neighbors_classifier.predict(x_test)

            new_object[target_classification] = predicted_class

        new_object = new_object.head(1)

    return new_object
