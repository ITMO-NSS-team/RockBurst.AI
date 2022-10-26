import pandas as pd

from sedef.preprocessing.cluster import \
    create_clusters, calculate_cluster_centroids
from sedef.preprocessing.grid import DiscreteStepSearch, \
    discretize_time_grid, merge_and_tens
from sedef.visualization.static import plot_3d_simple, plot_3d_clusters
from sedef.visualization.interactive import interactive_3d_simple, \
    interactive_line_plot

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('./datasets/cluster_example_dataset.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])

print('\n Source dataframe: ')
print(df.head(3))

# Create clusters for dataframe
df = create_clusters(dataframe=df,
                     min_cluster_size=10, min_samples=10,
                     list_columns=['X', 'Y', 'Z'],
                     new_column='Cluster')

# Plot static 3D plot for source data
plot_3d_simple(dataframe=df,
               columns_to_show={'x': 'X',
                                'y': 'Y',
                                'z': 'Z',
                                'target': 'Energy'},
               view_init=(60, 70))

# Plot interactive 3d
interactive_3d_simple(dataframe=df,
                      columns_to_show={'x': 'X',
                                       'y': 'Y',
                                       'z': 'Z',
                                       'target': 'Energy'})


# Calculate cluster centroids
df = calculate_cluster_centroids(dataframe=df,
                                 cluster_column='Cluster',
                                 list_columns=['X', 'Y', 'Z'])

print('\n Dataframe after clusterization and calculation clusters centroids')
print(df.head(3))

# Plot static 3D plot with clusters and cluster centroids
plot_3d_clusters(dataframe=df,
                 columns_to_show={'x': 'X',
                                  'y': 'Y',
                                  'z': 'Z',
                                  'cluster': 'Cluster'},
                 view_init=(60, 70))

# Finding the optimal sampling step using brute force search
discretizer = DiscreteStepSearch(df, collision_w=0.5, gaps_w=0.5, vis=False)
step_seconds = discretizer.time_step_search('Datetime', method='brute')

print(f'Optimal time step {step_seconds:.0f} sec.')
print(f'Optimal time step {(step_seconds/60):.0f} min.')

# Discretize time grid with optimal time step
df = discretize_time_grid(dataframe=df,
                          start_date='2019-02-01 00:00:00',
                          end_date='2019-02-10 00:00:00',
                          old_column='Datetime',
                          new_column='New_datetime',
                          time_step='51min')

print('\n Dataframe after sampling')
print(df.head(5))

# Make merge and tense operation for Energy column
df_merged = merge_and_tens(dataframe=df.copy(),
                           cluster_column='Cluster',
                           new_time_column='New_datetime',
                           old_time_column='Datetime',
                           list_columns=['X', 'Y', 'Z'],
                           target_regression='Energy',
                           target_classification=None)

print(df_merged.head(5))

# Plot interactive 2d
interactive_line_plot(dataframe=df_merged,
                      columns_to_show={'x': 'New_datetime',
                                       'y': 'Energy',
                                       'color': 'Cluster'})
