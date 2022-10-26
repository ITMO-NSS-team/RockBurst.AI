import numpy as np
import pandas as pd
from numpy import matrix as m
from pandas import DataFrame as df
from scipy import linalg


class Ssa:
    """Класс который создает новый объект содержащий в себе различную информацию
    о спектральном разложении методом SSA заданного временного ряда
    Parameters
    ----------
    time_series : pandas DataFrame
        датафрейм над которым требуется выполнить SSA разложение.Формат датафрейма:
        index-datetime,
        feature_col-int/float
    Returns
    -------
    python object
    """

    def __init__(self, time_series):

        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0]
        if self.ts_name == 0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.freq = self.ts.index.inferred_freq

    @staticmethod
    def _printer(name, *args):
        """Функция для принта"""
        print('-' * 40)
        print(name + ':')
        for msg in args:
            print(msg)

    @staticmethod
    def _dot(x, y):
        """Для работы с nan и неполными матрицами"""
        pass

    @staticmethod
    def get_contributions(X=None, s=None, plot=True):
        """Вычисляет значения вклада каждого сингулярного значения"""
        lambdas = np.power(s, 2)
        frob_norm = np.linalg.norm(X)
        ret = df(lambdas / (frob_norm ** 2), columns=['Contribution'])
        ret['Contribution'] = ret.Contribution.round(4)
        if plot:
            ax = ret[ret.Contribution != 0].plot.bar(legend=False)
            ax.set_xlabel("Lambda_i")
            ax.set_title('Non-zero contributions of Lambda_i')
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.2f}%'.format(x * 100) for x in vals])
            return ax
        return ret[ret.Contribution > 0]

    @staticmethod
    def diagonal_averaging(hankel_matrix):
        """Антидиагонализация Ганкелевой матрицы
        Returns: Pandas DataFrame object"""
        mat = m(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L, K), max(L, K)
        new = np.zeros((L, K))
        if L > K:
            mat = mat.T
        ret = []

        # Диагонализация
        for k in range(1 - K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star, :]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1 - mask)
            ret += [ma.sum() / mask_n]

        return df(ret).rename(columns={0: 'Reconstruction'})

    def view_time_series(self):
        """Визуализация временного ряда"""
        self.ts.plot(title='Original Time Series')

    def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):
        """Эмбединг временного ряда длины равной длине окна"""
        if not embedding_dimension:
            embedding_dimension = self.ts_N // 2
        else:
            embedding_dimension = embedding_dimension
        if suspected_frequency:
            suspected_frequency = suspected_frequency
            embedding_dimension = (embedding_dimension // suspected_frequency) * suspected_frequency

        lenght_window = self.ts_N - embedding_dimension + 1
        X = m(linalg.hankel(self.ts, np.zeros(embedding_dimension))).T[:, :lenght_window]
        X_df = df(X)
        X_complete = X_df.dropna(axis=1)
        self.X_com = m(X_complete.values)
        X_missing = X_df.drop(X_complete.columns, axis=1)
        X_miss = m(X_missing.values)
        trajectory_dimentions = X_df.shape
        self.complete_dimensions = X_complete.shape
        missing_dimensions = X_missing.shape
        no_missing = missing_dimensions[1] == 0

        if verbose:
            msg1 = 'Embedding dimension\t:  {}\nTrajectory dimensions\t: {}'
            msg2 = 'Complete dimension\t: {}\nMissing dimension     \t: {}'
            msg1 = msg1.format(embedding_dimension, trajectory_dimentions)
            msg2 = msg2.format(self.complete_dimensions, missing_dimensions)
            self._printer('EMBEDDING SUMMARY', msg1, msg2)

        if return_df:
            return X_df

    def decompose(self, verbose=False):
        """Выполняет SVD разложение и вычисляет ранк подпространства эмбединга
        Characteristic of projection: процент дисперсии захваченный в эмбединг"""
        X = self.X_com
        self.S = X * X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        Vs, Xs, Ys, Zs = {}, {}, {}, {}
        for i in range(self.d):
            Zs[i] = self.s[i] * self.V[:, i]
            Vs[i] = X.T * (self.U[:, i] / self.s[i])
            Ys[i] = self.s[i] * self.U[:, i]
            Xs[i] = Ys[i] * (m(Vs[i]).T)
        self.Vs, self.Xs = Vs, Xs
        self.s_contributions = self.get_contributions(X, self.s, False)
        self.r = len(self.s_contributions[self.s_contributions > 0])
        self.r_characteristic = round((self.s[:self.r] ** 2).sum() / (self.s ** 2).sum(), 4)
        self.orthonormal_base = {i: self.U[:, i] for i in range(self.r)}

        if verbose:
            msg1 = 'Rank of trajectory\t\t: {}\nDimension of projection space\t: {}'
            msg1 = msg1.format(self.d, self.r)
            msg2 = 'Characteristic of projection\t: {}'.format(self.r_characteristic)
            self._printer('DECOMPOSITION SUMMARY', msg1, msg2)

    def view_s_contributions(self, adjust_scale=False, cumulative=False, return_df=False):
        """Визуализация вклада каждого сингулярного значения и его сигнала"""
        contribs = self.s_contributions.copy()
        contribs = contribs[contribs.Contribution != 0]
        if cumulative:
            contribs['Contribution'] = contribs.Contribution.cumsum()
        if adjust_scale:
            contribs = (1 / contribs).max() * 1.1 - (1 / contribs)
        ax = contribs.plot.bar(legend=False)
        ax.set_xlabel("Singular_i")
        ax.set_title('Non-zero{} contribution of Singular_i {}'. \
                     format(' cumulative' if cumulative else '', '(scaled)' if adjust_scale else ''))
        if adjust_scale:
            ax.axes.get_yaxis().set_visible(False)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])
        if return_df:
            return contribs

    @classmethod
    def view_reconstruction(cls, *hankel, names=None, return_df=False, plot=True, symmetric_plots=False):
        """Восстановления исходного ряда по выбранным сингулярным значениям"""
        hankel_mat = None
        for han in hankel:
            if isinstance(hankel_mat, m):
                hankel_mat = hankel_mat + han
            else:
                hankel_mat = han.copy()
        hankel_full = cls.diagonal_averaging(hankel_mat)
        title = 'Reconstruction of signal'
        if names or names == 0:
            title += ' associated with singular value{}: {}'
            title = title.format('' if len(str(names)) == 1 else 's', names)
        if plot:
            ax = hankel_full.plot(legend=False, title=title)
            if symmetric_plots:
                velocity = hankel_full.abs().max()[0]
                ax.set_ylim(bottom=-velocity, top=velocity)
        if return_df:
            return hankel_full

    def _forecast_prep(self, singular_values=None):
        self.X_com_hat = np.zeros(self.complete_dimensions)
        verticality_coefficient = 0
        forecast_orthonormal_base = {}
        if singular_values:
            try:
                for i in singular_values:
                    forecast_orthonormal_base[i] = self.orthonormal_base[i]
            except:
                if singular_values == 0:
                    forecast_orthonormal_base[0] = self.orthonormal_base[0]
                else:
                    raise ('Please pass in a list/array of singular value indices to use for forecast')
        else:
            forecast_orthonormal_base = self.orthonormal_base
        self.R = np.zeros(forecast_orthonormal_base[0].shape)[:-1]
        for Pi in forecast_orthonormal_base.values():
            self.X_com_hat += Pi * Pi.T * self.X_com
            pi = np.ravel(Pi)[-1]
            verticality_coefficient += pi ** 2
            self.R += pi * Pi[:-1]
        self.R = m(self.R / (1 - verticality_coefficient))
        self.X_com_tilde = self.diagonal_averaging(self.X_com_hat)

    def forecast_recurrent(self, steps_ahead=12, singular_values=None, plot=False, return_df=False, **plotargs):
        """Метод для прогноза/заполнения пропущенных значения рекурретным методом"""
        try:
            self.X_com_hat
        except AttributeError:
            self._forecast_prep(singular_values)
        ts_forecast = np.array(self.ts_v[0])
        for i in range(1, self.ts_N + steps_ahead):
            try:
                if np.isnan(self.ts_v[i]):
                    x = self.R.T * m(ts_forecast[max(0, i - self.R.shape[0]): i]).T
                    ts_forecast = np.append(ts_forecast, x[0])
                else:
                    ts_forecast = np.append(ts_forecast, self.ts_v[i])
            except(IndexError):
                x = self.R.T * m(ts_forecast[i - self.R.shape[0]: i]).T
                ts_forecast = np.append(ts_forecast, x[0])
        forecast_N = i + 1
        new_index = pd.date_range(start=self.ts.index.min(), periods=forecast_N, freq=self.freq)
        forecast_df = df(ts_forecast, columns=['Forecast'], index=new_index)
        forecast_df['Original'] = np.append(self.ts_v, [np.nan] * steps_ahead)
        if plot:
            forecast_df.plot(title='Forecasted vs. original time series', **plotargs)
        if return_df:
            return forecast_df
