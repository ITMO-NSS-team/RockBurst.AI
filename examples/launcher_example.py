import numpy as np
import pandas as pd

from sedef.launch import Launcher

df = pd.read_csv('./datasets/cluster_example_dataset.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Parameters for simple visualisations
params_3d_vis = {'columns_to_show': {'x': 'X', 'y': 'Y',
                                     'z': 'Z', 'target': 'Energy'},
                 'view_init': (60, 70)}

# Parameters for clusterization
params_for_clustering = {'min_cluster_size': 10,
                         'min_samples': 10,
                         'list_columns': ['X', 'Y', 'Z']}

# Parameters for optimal time step search
params_for_search = {'collision_w': 0.5,
                     'gaps_w': 0.5,
                     'vis': True,
                     'method': 'brute'}

# Parameters for optimal time step setting
params_for_setting = {'start_date': '2019-02-01 00:00:00',
                      'end_date': '2019-02-10 00:00:00',
                      'time_step': '1H'}

# Parameters for merge operation
params_for_merge = {'list_columns': ['X', 'Y', 'Z'],
                    'target_regression': 'Energy',
                    'target_classification': None}

# Parameters for line plot
params_line_plot = {'columns_to_show': {'x': 'New_datetime',
                                        'y': 'Energy'}}

# Configuration as dict
configuration = {'3D show static': params_3d_vis,
                 '3D show interactive': params_3d_vis,
                 'Create clusters with centroids': params_for_clustering,
                 '3D show clusters static': params_3d_vis,
                 'Find optimal time step': params_for_search,
                 'Set time step': params_for_setting,
                 'Merge and tense': params_for_merge,
                 'Line show interactive': params_line_plot}

launcher = Launcher(dataframe=df, datetime_column='Datetime')
# Run the configuration
final_dataframe = launcher.run(configuration=configuration)

print(final_dataframe.head(10))
