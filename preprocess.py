from multiprocessing.sharedctypes import Value
from typing import Any

import numpy as np
import pandas as pd
from clearml import Task
import pickle
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        self.task = Task.get_task(project_name='serving examples', task_id='bfc1ae4d242b4d5a9955adde1c9e5a58')
        if self.task.artifacts.get('seasonality').size is not None:
            self.seasonality = eval(self.task.artifacts.get('seasonality').preview)
            self.z_score = eval(self.task.artifacts.get('z_score').preview)
        else:
            self.z_score = None

        self.threshold = eval(self.task.artifacts.get('threshold').preview)
        self.timesteps = eval(self.task.artifacts.get('time_steps').preview)
        self.trend_step = eval(self.task.artifacts.get('trend_step').preview)

        regressor_path = self.task.artifacts['regressor'].get_local_copy()
        file_regressor = open(regressor_path, 'rb')
        self.regressor = pickle.load(file_regressor)
        pass

    def z_score_normalization(self, df,seasonality, normal_z_score=None):
        indexer = 0
        if seasonality == 'hourly':
            season = df.groupby(df.index.hour)
        if seasonality == 'daily':
            season = df.groupby(df.index.dayofweek)
        if seasonality == 'weekly':
            season = df.groupby(df.index.isocalendar().week)
            indexer = 1
        if seasonality == 'monthly':
            season = df.groupby(df.index.month)
            indexer = 1

        if normal_z_score is not None:
            for index, time in season:
                index -= indexer
                df.loc[time.index, 'value'] = (time['value'] - normal_z_score[index]['mean'])/normal_z_score[index]['std']
            return df
        else:  
            z_score = []
            for _, time in season:
                mean = np.mean(time['value'])
                std = np.std(time['value'])
                df.loc[time.index, 'value'] = (time['value'] - mean)/std
                z_score.append({'mean': mean, 'std': std})
            return df, z_score

    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X)-self.timesteps):
            Xs.append(X.iloc[i:(i+self.timesteps)].values)
            ys.append(y.iloc[i+self.timesteps])
        
        return np.array(Xs), np.array(ys)

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        df = pd.DataFrame({'date': body.get("date"), 'value': body.get("value")}, columns=['date','value'])
        
        df['date'] = pd.to_datetime(df['date'], dayfirst = True)
        df = df.set_index('date')

        df = df.resample('1T').mean()
        # detrend regression here
        y_test_value = df['value'].values
        X_test_value = [i + self.trend_step for i in range(0, len(df['value']))]
        X_test_value = np.reshape(X_test_value, (len(X_test_value), 1))
        # X_test_poly = preprocessing.get_polynomial_features(X_test_value)
        trendp = self.regressor.predict(X_test_value)
        detrended = [y_test_value[i]-trendp[i] for i in range(0, len(df['value']))]
        df['value'] = detrended
        self.task.upload_artifact('trend_step', self.trend_step + 1)
                
        #ambil dari artifact seasonal dan z_scorenya, diff, threshold
        if self.z_score is not None:
            df = self.z_score_normalization(df, self.seasonality, self.z_score)

        X, _ = self.create_sequences(df[['value']],  df['value'])
        state['X'] = X
        state['df_test'] = df
        return X

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        
        X_test = state['X']
        df_test = state['df_test']
        
        test_mae_loss = np.mean(np.abs(data-X_test), axis=1)
        test_score_df = pd.DataFrame(df_test[self.timesteps:])
        test_score_df['loss'] = test_mae_loss
        test_score_df['threshold'] = self.threshold
        test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
        test_score_df['Close'] = df_test[self.timesteps:]['value']

        # raise ValueError("until here is fine")
        return test_score_df.to_dict()
