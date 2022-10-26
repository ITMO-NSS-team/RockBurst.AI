import numpy as np
import pandas as pd

from sedef.preprocessing.cluster import \
    create_clusters, calculate_cluster_centroids
from sedef.preprocessing.grid import DiscreteStepSearch, \
    discretize_time_grid, merge_and_tens
from sedef.visualization.static import plot_3d_simple, plot_3d_clusters
from sedef.visualization.interactive import interactive_3d_simple, \
    interactive_line_plot


class Launcher:
    """
    Class for running the processing strategy in one line (or small amount of
    lines) of code.

    The main argument when initializing the class is a dictionary with the
    listed data processing blocks.

    Available data processing blocks:
    'Create clusters with centroids'
    'Set time step'
    'Merge and tense'
    'SSA'
    'Find optimal time step' - [optional]

    Blocks for visualizations:
    '3D show static' - [optional]
    '3D show clusters static' - [optional]
    '3D show interactive' - [optional]
    'Line show interactive' - [optional]
    TODO add new features as blocks
    """

    _order = ['3D show static', '3D show interactive',
              'Create clusters with centroids',
              '3D show clusters static', 'Find optimal time step',
              'Set time step', 'Merge and tense', 'Line show interactive']

    _optional = ['Find optimal time step', '3D show static',
                 '3D show clusters static', '3D show interactive',
                 'Line show interactive']

    _forbidden_names = ['Cluster', 'New_datetime', 'distance']

    def __init__(self, dataframe: pd.DataFrame, datetime_column: str):
        self.df = dataframe
        self.datetime_column = datetime_column
        self.auto_time_step = None
        self.time_step = None

    def run(self, configuration: dict):
        """ Running the process """

        # Check correctness of dataframe column names
        self._check_df(configuration)

        # Check the correctness of chain ang get ids of operations in order list
        blocks_ids = self._check_validity(configuration)

        # Run the process
        for blocks_id in blocks_ids:
            operation_name = self._order[blocks_id]
            operation_params = configuration.get(operation_name)

            # Create an object for define appropriate methods
            controller = BlocksFabrid(dataframe=self.df,
                                      auto_time_step=self.auto_time_step)
            if operation_name == 'Find optimal time step':
                time_step = controller.do_process(operation_name=operation_name,
                                                  params=operation_params,
                                                  datetime_column=self.datetime_column,
                                                  time_step=self.time_step)
                self.time_step = time_step
            else:
                controller.do_process(operation_name=operation_name,
                                      params=operation_params,
                                      datetime_column=self.datetime_column,
                                      time_step=self.time_step)

            # Update dataframe after processing
            self.df = controller.get_dataframe()

        return self.df

    def _check_df(self, configuration):
        """ Method check correctness of column names in dataframe """
        processing_blocks = list(configuration.keys())

        for name in self.df.columns:
            if name == 'Cluster' and any('Create clusters with centroids' != block for block in processing_blocks):
                pass
            else:
                if any(name == bad_name for bad_name in self._forbidden_names):
                    ex = f'Column name "{name}" is forbidden change name to start'
                    raise ValueError(ex)

    def _check_validity(self, configuration):
        """ Method check was an obligatory blocks skipped or not """
        processing_blocks = list(configuration.keys())
        processing_blocks_arr = np.array(processing_blocks)
        order_arr = np.array(self._order)

        block_1 = 'Set time step'
        block_2 = 'Find optimal time step'
        auto_time_step = False
        block_1_id = np.argwhere(processing_blocks_arr == block_1)
        block_2_id = np.argwhere(processing_blocks_arr == block_2)
        if len(block_1_id) != 0 and len(block_2_id) != 0:
            print(f'Warning! During the "{block_2}" operation there will be '
                  f'defined optimal time step, so operation "{block_1}" with '
                  f'pre-defined parameter time_step will be ignored')
            auto_time_step = True

        self.auto_time_step = auto_time_step

        user_orders = []
        for block in processing_blocks:
            index_by_order = int(np.argwhere(order_arr == block)[0])
            user_orders.append(index_by_order)

        # Get ids of obligatory blocks till the last index
        for i in range(0, max(user_orders)+1):
            operation = order_arr[i]
            if any(operation == optional for optional in self._optional):
                pass
            elif any(operation == defined for defined in processing_blocks):
                pass
            else:
                ex = f'An obligatory processing block "{operation}" was skipped'
                raise AttributeError(ex)

        # Ids of operations defined by user
        user_orders.sort()
        return user_orders


class BlocksFabrid:
    """
    Class for providing appropriate operations by it's name and parameters
    """
    def __init__(self, dataframe, auto_time_step):
        self.dataframe = dataframe
        self.auto_time_step = auto_time_step

    def do_process(self, operation_name, params,
                   datetime_column, time_step):

        if operation_name == '3D show static':
            # OPTIONAL
            plot_3d_simple(dataframe=self.dataframe,
                           columns_to_show=params.get('columns_to_show'),
                           view_init=params.get('view_init'))

        elif operation_name == '3D show interactive':
            # OPTIONAL
            interactive_3d_simple(dataframe=self.dataframe,
                                  columns_to_show=params.get('columns_to_show'))

        elif operation_name == 'Create clusters with centroids':
            # Create clusters for dataframe
            df = create_clusters(dataframe=self.dataframe,
                                 min_cluster_size=params.get('min_cluster_size'),
                                 min_samples=params.get('min_samples'),
                                 list_columns=params.get('list_columns'),
                                 new_column='Cluster')

            # Calculate cluster centroids
            df = calculate_cluster_centroids(dataframe=df,
                                             cluster_column='Cluster',
                                             list_columns=params.get('list_columns'))
            # Update dataframe
            self.dataframe = df

        elif operation_name == '3D show clusters static':
            # OPTIONAL
            updated = params.get('columns_to_show')
            updated.update({'cluster': 'Cluster'})
            params.update({'columns_to_show': updated})

            plot_3d_clusters(dataframe=self.dataframe,
                             columns_to_show=params.get('columns_to_show'),
                             view_init=params.get('view_init'))

        elif operation_name == 'Find optimal time step':
            discretizer = DiscreteStepSearch(self.dataframe,
                                             collision_w=params.get('collision_w'),
                                             gaps_w=params.get('gaps_w'),
                                             vis=params.get('vis'))
            step_seconds = discretizer.time_step_search(datetime_column,
                                                        method=params.get('method'))
            # Use optimal time step
            str_step_seconds = str(int(round(step_seconds)))
            time_step = ''.join((str_step_seconds, 's'))
        elif operation_name == 'Set time step':
            if self.auto_time_step:
                df = discretize_time_grid(dataframe=self.dataframe,
                                          start_date=params.get('start_date'),
                                          end_date=params.get('end_date'),
                                          old_column=datetime_column,
                                          new_column='New_datetime',
                                          time_step=time_step)
            else:
                df = discretize_time_grid(dataframe=self.dataframe,
                                          start_date=params.get('start_date'),
                                          end_date=params.get('end_date'),
                                          old_column=datetime_column,
                                          new_column='New_datetime',
                                          time_step=params.get('time_step'))

            # Update dataframe
            self.dataframe = df

        elif operation_name == 'Merge and tense':
            df = merge_and_tens(dataframe=self.dataframe,
                                cluster_column='Cluster',
                                new_time_column='New_datetime',
                                old_time_column=datetime_column,
                                list_columns=params.get('list_columns'),
                                target_regression=params.get('target_regression'),
                                target_classification=params.get('target_classification'))

            # Update dataframe
            self.dataframe = df

        elif operation_name == 'Line show interactive':
            # OPTIONAL
            updated = params.get('columns_to_show')
            updated.update({'color': 'Cluster'})
            params.update({'columns_to_show': updated})
            interactive_line_plot(dataframe=self.dataframe,
                                  columns_to_show=params.get('columns_to_show'))

        return time_step

    def get_dataframe(self):
        return self.dataframe
