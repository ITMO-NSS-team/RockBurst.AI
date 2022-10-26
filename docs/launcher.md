# Launcher submodule

## Short description

To make it easier to run all the functions in one pipeline, a high-level api
 was implemented that allows you to build and run pipelines based on blocks in the form of dictionaries.
 
____________

## Launcher class

#### Arguments

    'dataframe' (pandas DataFrame) - the dataframe to process
    'datetime_column' (str) - name of the column that is present in the datset and contains the datetime
    
### run method

A methods run the process, defined in configuration.

#### Parameters

    'configuration' (dict) - dictionary with a list of blocks and their parameters
    
#### Returns
    
    'df' (pandas DataFrame) - resulting dataframe
    
For more information check [tutorial](https://github.com/ITMO-NSS-team/SeismicDeformation/blob/main/examples/launcher_tutorial.ipynb) (in russian).