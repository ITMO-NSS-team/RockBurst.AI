import pandas as pd
from sedef.analysis.ssa_utils import Ssa


class SsaDecompose:
    """Класс который создает новый объект содержащий в себе различную информацию
    о спектральном разложении методом SSA заданного временного ряда
    Parameters
    ----------
    seismic_df : pandas DataFrame
        датафрейм над которым требуется выполнить SSA разложение.Формат датафрейма:
        index-datetime,
        feature_col-int/float
    time_slice : int
        модерируемый параметр, устанавливает отсечку по времени, по умолчанию равен -1
    seismic_field : str
        имя колонки с данными случайной величины над которой выполняется преобразование
    time_field: str
        имя колонки в которой содержатся временные отсечки
    mean_field: str
        имя колонки в которую записываются среднее значений за выбранный период
    sum_field: str
        имя колонки в которую записываются сумма значений за выбранный период
    Returns
    -------
    python object
    """

    def __init__(self,
                 df: pd.DataFrame,
                 time_slice: int = -1,
                 feature_field: str = 'Энергия',
                 time_field: str = 'Время',
                 mean_field: str = 'Среднее значение величины сейсмических событий за день',
                 sum_field: str = 'Сумма всех значений величин сейсмических событий за день'):
        """
Initializes the class with the provided parameters.

    This constructor method sets up the instance variables required for further processing of the data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be processed.
        time_slice (int): The number of rows to consider for the time series analysis.
        feature_field (str): The name of the feature field in the DataFrame.
        time_field (str): The name of the time field in the DataFrame, which will be converted to datetime.
        mean_field (str): The name of the field used for calculating the mean.
        sum_field (str): The name of the field used for calculating the sum.

    Returns:
        None
    """
        self.df = df
        self.time_slice = time_slice
        self.feature_field = feature_field
        self.time_field = time_field
        self.mean_field = mean_field
        self.sum_field = sum_field

    def preprocess(self, sum_flag: bool = True):
        """
Preprocess the time series data by calculating the sum or mean.

    This method processes the time series data stored in the instance variable `ts`. 
    Depending on the value of `sum_flag`, it either calculates the sum or the mean 
    of the specified feature field grouped by date, month, and year. The resulting 
    values are stored in a new column 'Sum_at_day'. Duplicate entries based on 
    the date, month, and year are then removed from the DataFrame.

    Args:
        sum_flag (bool): A flag indicating whether to calculate the sum (True) 
                         or the mean (False) of the feature field.

    Returns:
        None: This method modifies the instance's time series data in place 
              and does not return any value.
    """
        self.df[self.time_field] = pd.to_datetime(self.df[self.time_field], dayfirst=True)
        self.ts = self.df.iloc[:self.time_slice]
        self.ts.set_index(self.time_field, inplace=True)
        self.ts['year'] = self.ts.index.strftime('%Y')
        self.ts['month'] = self.ts.index.strftime('%b')
        self.ts['date'] = self.ts.index.strftime('%d')
        self.ts['hour'] = self.ts.index.strftime('%H')
        self._make_conj(self.ts, 'date', 'month', 'year')
        if sum_flag:
            self.ts['Sum_at_day'] = self._code_sum(self.ts, 'date + month + year', self.feature_field).astype(int)
        else:
            self.ts['Sum_at_day'] = self._code_mean(self.ts, 'date + month + year', self.feature_field).astype(int)
        self.ts = self.ts.drop_duplicates(subset=['date + month + year'], keep='first', inplace=False)
        return self.ts

    def ssa_report(self, suspected_dimension, suspected_seasonality):
        """
Generates a report based on Singular Spectrum Analysis (SSA) for the given suspected dimension and seasonality.

    This method analyzes the time series data using SSA techniques to identify patterns and characteristics 
    related to the specified dimension and seasonality. It does not return any value but may produce visual 
    outputs or save reports as a side effect.

    Args:
        suspected_dimension (int): The suspected dimension to be analyzed in the SSA.
        suspected_seasonality (int): The suspected seasonality to be considered in the SSA analysis.

    Returns:
        None: This method does not return any value.
    """
        ssa_ts = self.preprocess()
        self.ssa_model = Ssa(ssa_ts[['Sum_at_day']])
        self.ssa_model.embed(embedding_dimension=suspected_dimension, suspected_frequency=suspected_seasonality,
                             verbose=True)
        self.ssa_model.decompose(verbose=True)
        return

    def ssa_contrib(self, ):
        """
Updates the time series model with reconstructed values.

    This method copies the time series from the `ssa_model` and updates it 
    with the reconstructed values.

    Args:
        self: The instance of the class that contains the `ssa_model` attribute.

    Returns:
        None
    """
        return self.ssa_model.view_s_contributions()

    def plot_ssa_values(self, num_values):
        """
Plot the Singular Spectrum Analysis (SSA) values.

    This method generates a plot comparing the original time series with its reconstructed version 
    based on the specified number of SSA components.

    Args:
        num_values (int): The number of SSA components to use for reconstruction.

    Returns:
        None
    """
        for i in range(num_values):
            Ssa.view_reconstruction(self.ssa_model.Xs[i], names=i, symmetric_plots=i != 0)
        return

    def plot_ssa_reconstructed(self, num_values):
        """
Plots the reconstructed SSA (Singular Spectrum Analysis) of a time series.

    This method visualizes the reconstructed components of a time series using Singular Spectrum Analysis.
    It generates a plot that displays the specified number of reconstructed values from the time series.

    Args:
        num_values (int): The number of reconstructed values to display in the plot.

    Returns:
        None: This method does not return any value. It generates a plot as a side effect.
    """
        streams10 = [i for i in range(num_values)]
        reconstructed10 = self.ssa_model.view_reconstruction(*[self.ssa_model.Xs[i] for i in streams10],
                                                             names=streams10, return_df=True, plot=False)
        ts_copy10 = self.ssa_model.ts.copy()
        ts_copy10['Reconstruction'] = reconstructed10.Reconstruction.values
        ts_copy10.plot(title='Original vs. Reconstructed Time Series')
        return

    def plot_time_series(self):
        """
Plot a time series based on categorical and real features.

    This method generates a time series plot by aggregating the values of a 
    specified real feature based on the categories defined in a specified 
    categorical feature. The aggregation is performed using the sum of the 
    real feature values for each category.

    Args:
        cat_feature (str): The name of the categorical feature used for grouping.
        real_feature (str): The name of the real feature whose values will be summed.

    Returns:
        None: This method does not return any value. It produces a plot as a side effect.
    """
        return self.df.plot.line(y=self.feature_field, x=self.time_field, figsize=(20, 4))

    @staticmethod
    def _make_conj(data, feature1, feature2, feature3):
        """
Constructs a conjunction of specified features from the given data.

    This method processes the input data to create a conjunction of the specified features,
    which can be used for further analysis or modeling.

    Args:
        data (iterable): The input data from which features will be extracted.
        feature1 (str): The first feature to include in the conjunction.
        feature2 (str): The second feature to include in the conjunction.
        feature3 (str): The third feature to include in the conjunction.

    Returns:
        None: This method does not return a value. It modifies the data in place or performs
        operations based on the conjunction of the specified features.
    """
        data[feature1 + ' + ' + feature2 + ' + ' + feature3] = data[feature1].astype(str) + ' + ' + data[
            feature2].astype(str) + ' + ' + data[feature3].astype(str)
        return data

    @staticmethod
    def _code_mean(data, cat_feature, real_feature):
        """
Calculates the mean of a real feature grouped by a categorical feature.

    This method computes the mean value of the specified real feature for each unique 
    category in the specified categorical feature. The results can be used for further 
    analysis or encoding of categorical variables.

    Args:
        data (pandas.DataFrame): The input data containing both categorical and real features.
        cat_feature (str): The name of the categorical feature to group by.
        real_feature (str): The name of the real feature for which the mean will be calculated.

    Returns:
        None: This method does not return a value. It may modify the input data or produce 
        side effects such as printing results or updating a data structure.
    """
        return data[cat_feature].map(data.groupby(cat_feature)[real_feature].mean())

    @staticmethod
    def _code_sum(data, cat_feature, real_feature):
        """
Calculates the sum of a specified real feature grouped by a categorical feature.

    This method processes the input data to compute the sum of values in the 
    specified real feature, categorized by the unique values in the specified 
    categorical feature. The results are typically used for further analysis 
    or reporting.

    Args:
        data (pandas.DataFrame): The input data containing both categorical and 
            real features.
        cat_feature (str): The name of the categorical feature to group by.
        real_feature (str): The name of the real feature for which the sum 
            will be calculated.

    Returns:
        None: This method does not return a value. It modifies the input data 
        or produces side effects as needed.
    """
        return data[cat_feature].map(data.groupby(cat_feature)[real_feature].sum())
