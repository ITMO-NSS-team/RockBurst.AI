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
        self.df = df
        self.time_slice = time_slice
        self.feature_field = feature_field
        self.time_field = time_field
        self.mean_field = mean_field
        self.sum_field = sum_field

    def preprocess(self, sum_flag: bool = True):
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
        ssa_ts = self.preprocess()
        self.ssa_model = Ssa(ssa_ts[['Sum_at_day']])
        self.ssa_model.embed(embedding_dimension=suspected_dimension, suspected_frequency=suspected_seasonality,
                             verbose=True)
        self.ssa_model.decompose(verbose=True)
        return

    def ssa_contrib(self, ):
        return self.ssa_model.view_s_contributions()

    def plot_ssa_values(self, num_values):
        for i in range(num_values):
            Ssa.view_reconstruction(self.ssa_model.Xs[i], names=i, symmetric_plots=i != 0)
        return

    def plot_ssa_reconstructed(self, num_values):
        streams10 = [i for i in range(num_values)]
        reconstructed10 = self.ssa_model.view_reconstruction(*[self.ssa_model.Xs[i] for i in streams10],
                                                             names=streams10, return_df=True, plot=False)
        ts_copy10 = self.ssa_model.ts.copy()
        ts_copy10['Reconstruction'] = reconstructed10.Reconstruction.values
        ts_copy10.plot(title='Original vs. Reconstructed Time Series')
        return

    def plot_time_series(self):
        return self.df.plot.line(y=self.feature_field, x=self.time_field, figsize=(20, 4))

    @staticmethod
    def _make_conj(data, feature1, feature2, feature3):
        data[feature1 + ' + ' + feature2 + ' + ' + feature3] = data[feature1].astype(str) + ' + ' + data[
            feature2].astype(str) + ' + ' + data[feature3].astype(str)
        return data

    @staticmethod
    def _code_mean(data, cat_feature, real_feature):
        return data[cat_feature].map(data.groupby(cat_feature)[real_feature].mean())

    @staticmethod
    def _code_sum(data, cat_feature, real_feature):
        return data[cat_feature].map(data.groupby(cat_feature)[real_feature].sum())
