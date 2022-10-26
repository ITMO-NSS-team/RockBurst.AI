# Preprocessing grid submodule

## Short description
preprocessing/grid - a set of methods that are needed for discretizing functions in space and time.
 This functions allows you to place scattered points in linked clusters across space and at regular 
 time intervals.

![discretize.png](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/images/discretize.png?raw=true)

As you can see from the image, a common problem for data supplied to this module 
is that the points do not intersect with each other. It is impossible to build a 
time series based on them even with an irregular step. Grid submodule can solve this problem.

____________

## DiscreteStepSearch class

The class is intended for selecting the optimal time step. The algorithm finds 
a balance between gaps (which are formed as a result of transformations) and collisions 
(when there are several values per timestamp).

#### Arguments

    'dataframe' (pandas DataFrame) - the dataframe to process
    'collision_w' (float, default = 0.5) - weight of the collision case when averaging (collision_w + gaps_w = 1)
    'gaps_w' (float, default = 0.5) - weight of the gap case when averaging (collision_w + gaps_w = 1)
    'vis' (bool, default = False) - is it necessary to visualize the considered solutions
    
### timestep_search method

A function that allows you to determine the optimal time step for sampling 
based on golden section method or brute force search.

#### Parameters

    'datetime_column' (str, default='Datetime') - name of the column that is present in the datset and contains the datetime
    'method' (str, default = 'golden') - name of method for solving optimization task - available methods are 'golden' and 'brute'.
    
#### Returns
    
    'step' (float) - suggested optimal time step in seconds
____________

## discretize_time_grid function

If all objects in the table have different time bindings and the timestamp intervals are not constant, then we need to place the values on a regular time grid. This function allows us to do this with a given sampling step.

Each row in the table is assigned a new timestamp. New time indices are assigned on a regular grid. Objects will be assigned new time index value based on the rule that if the old timestamp >= new timestamp - 'timestep'/2 and < new timestamp + 'timestep'/2.

#### Parameters

    'dataframe' (pandas DataFrame) - the dataframe to process
    'start_date' (str) - date time of the start of the time interval for sampling in the format " %Y - %m-%d %H:%M:%S"
    'end_date' (str) - date time of the end of the time interval for sampling in the format " %Y - %m-%d %H:%M:%S"
    'old_column' (str, default='Datetime') - name of the column that is present in the dataset and contains the date/time
    'new_column' (str, default='New_datetime') - the column that is needed to create
    'time_step' (str, default='1H') - the time step that you want to bring the series to. For example, '5min' - 5 minutes, '5H' - 5 hours, '5D' - 5 days, etc.

    
#### Returns
    
    'new_dataframe' (pandas DataFrame) - dataframe with an additional new_column column that contains equidistant time series labels
    

____________


## merge_and_tens function 

At this stage, the table should already have columns with assigned cluster labels and new regular 
time indices. This function allows you to "drag" values in points to new stable ones. 
At the same time, we need to evaluate the values of some variables (continuous or categorical) 
in these nodes.

This task is performed using a sequence of actions:
* Calculation of cluster centroid coordinates - X, Y, Z;
* For each time index on a regular grid, a set of points (rows from the table) is defined. These points belong to the same time period, but may belong to different spatial clusters. Moreover, the initial timestamps for these points may (and definitely will) also differ;
* So at this stage, we know exactly the coordinates of the cluster centroid and the value of the time index for these objects. Therefore, we calculated the difference for each coordinate (X,Y,Z and time) in the form: "Distance by X = X coordinate of the cluster centroid - the X coordinate of this object". The time for this operation is converted to absolute values for example, in minutes within a given time block;
* Based on the obtained distances, the value of the target variable in the center of the cluster is predicted for the new timestamp using the K-nearest neighbor algorithm, which passes the distance values for all four coordinates as features;
* Creating a dataset in which each coordinate of the cluster centroid is associated with a certain value of the target variable at a certain time index.

#### Parameters

    'dataframe' (pandas DataFrame) - the dataframe to process, the dataframe must contain columns 'X', 'Y', 'Z' with coordinates
    'cluster_column' (str) - name of the column that contains cluster labels
    'new_time_column' (str) - name of the column that contains equidistant datetime values
    'old_time_column' (str) - name of the column that contains non-equidistant (original) datetime values
    'list_columns' (str) - list with names of columns with features, which were used to predict cluster labels
    'target_regression' (str, default = None) - name of the column with the continuous target variable
    'target_classification' (str, default = None) - name of the column with the target categorical variable

    
#### Returns
    
    'new_dataframe' (pandas DataFrame) - Dataframe with sampled values in time and space (clusters)

As a result of these actions, an array of points located on a regular grid will be obtained, both in space and in time. A simplified demonstration of the algorithm can be seen below

![grid_animation.gif](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/images/grid_animation.gif?raw=true)

## Examples


```python
from sedef.preprocessing.grid import DiscreteStepSearch

# Finding the optimal sampling step
discretizer = DiscreteStepSearch(dataframe=df, 
                                 collision_w = 0.5, 
                                 gaps_w = 0.5, 
                                 vis=False)
step = discretizer.timestep_search('Datetime')
```
    


```python
from sedef.preprocessing.grid import discretize_time_grid

# discretize_time_grid
df = discretize_time_grid(dataframe=df,
                          start_date='2019-02-01 00:00:00',
                          end_date='2019-02-10 00:00:00',
                          old_column='Datetime',
                          new_column='New_datetime',
                          time_step='51min')
```



```python
from sedef.preprocessing.grid import merge_and_tens

# merge_and_tens
df_merged = merge_and_tens(dataframe=df,
                           cluster_column='Cluster',
                           new_time_column='New_datetime',
                           old_time_column='Datetime',
                           list_columns=['X', 'Y', 'Z'],
                           target_regression='Energy',
                           target_classification=None)
```

