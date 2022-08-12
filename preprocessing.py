import pandas as pd
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RANSACRegressor

def read_data(path, datetime, metric_name, metric_type, node_instance):
    with open(path) as data_file:
        data = json.load(data_file)
        df = pd.json_normalize(data, ['data', 'meta_data'], max_level=1)
    #filter dataframe code here

scaler = MinMaxScaler()
def scale(df, is_test = False):
    if not is_test:
        df = scaler.fit_transform(df)
    else:
        df = scaler.transform(df)
    return df

def difference(data, column):
    previous_diff = data[column]
    counter = 0
    # buatcheck p-value tapi not work
    index, res = None, None
    while index is None:
        if previous_diff.equals(data[column]) is False:
            previous_diff = data[column]
        data[column] = data[column].diff()
        data = data.dropna()
        index, res = check_acf_plot(data[column])
        if index == 1 and res[index] < -0.5:
            return previous_diff
        elif index is not None:
            return data
        counter += 1
        print(f'order of difference: {counter}')
    return data

def check_p_value(data):
    result = adfuller(data.dropna())
    print(f'p-value: {result[1]}')
    if result[1] < 0.05:
        return True
    return False

def check_acf_plot(data):
    res,confint, qstat, pvalues = acf(data, nlags=100, qstat=True, alpha=.05)
    for index, i in enumerate(res[:10]):
        if i<0:
            print(f'found negative ac at lag-{index}')
            return index, res
    return None, res

def get_seasonality(df):
    mean_std_list = {}
    season = df.groupby(df.index.hour)
    mean_std_list['hourly'] = get_mean_std(season)
    previous = mean_std_list['hourly']
    seasonality = 'hourly'

    season = df.groupby(df.index.weekday)
    mean_std_list['daily'] = get_mean_std(season)

    season = df.groupby(df.index.isocalendar().week)
    mean_std_list['weekly'] = get_mean_std(season)

    season = df.groupby(df.index.month)
    mean_std_list['monthly'] = get_mean_std(season)

    for key, value in mean_std_list.items():
        temp = value
        if temp < previous:
            seasonality = key
            previous = temp
    df_temp = df.copy(deep=True)
    df_temp, _ = z_score(df_temp,seasonality=seasonality)
    if df_temp['value'].std() < df['value'].std():
        print(f'seasonality: {seasonality}')
        return seasonality
    else:
        print(f'no need seasonality')
        return None

def get_mean_std(season):
    return np.mean([np.std(time['value']) for index, time in season])

def get_minimum_lag(df):
    res,confint, qstat, pvalues = acf(df['value'], nlags=int(len(df) * 0.25), qstat=True, alpha=.05)
    for i, (x_acf, conf) in enumerate(zip(res, confint[:, 1])):
        if x_acf < conf - x_acf:
            diff_period = i
            return diff_period
    return None

def z_score(df,seasonality, normal_z_score=None):
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
        z_score = {}
        for index, time in season:
            # median = np.median(time['value'])
            # deviation_from_med = np.array(time['value']) - median
            # mad = np.median(np.abs(deviation_from_med))
            # consistency_correction = 1.4826
            # df.loc[time.index, 'value'] = deviation_from_med/(consistency_correction*mad)
            mean = np.mean(time['value'])
            std = np.std(time['value'])
            df.loc[time.index, 'value'] = (time['value'] - mean)/std
            z_score[index] = {'mean': mean, 'std': std}
        return df, z_score

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

def moving_average(df, column, window):
    df[[column]] = df[[column]].rolling(window, center=True).mean()
    df=df.dropna()
    return df

def rmse (actual,pred):
    actual,pred = np.array(actual), np.array(pred)
    return np.sqrt(((pred - actual) ** 2).mean())

def detrend_regression(df):
    X_test_value = [i for i in range(0, len(df['value']))]
    X_test_value = np.reshape(X_test_value, (len(X_test_value), 1))
    y_test_value = df['value'].values

    linear_regressor = create_ransac_model()
    # polynomial_regressor = create_ransac_model()
    # X_poly_value = get_polynomial_features(X_test_value)

    print(type(y_test_value))
    y_new = []
    for item in y_test_value:
        if str(item) != 'nan':
            y_new.append(item)

    x_new = np.delete(X_test_value, 0)
    print(type(x_new))
    np.reshape(x_new, )
    print(x_new.shape)

    # linear_regressor.fit(X_test_value, y_test_value)
    linear_regressor.fit(x_new, y_new)
    # polynomial_regressor.fit(X_poly_value, y_test_value)

    linear_trend = linear_regressor.predict(X_test_value)
    # polynomial_trend = polynomial_regressor.predict(X_poly_value)

    # if rmse(y_test_value, linear_trend) <= rmse(y_test_value, polynomial_trend):
    #     print(f'Linear Regression Choosed')
    detrended = [y_test_value[i]-linear_trend[i] for i in range(0, len(df['value']))]
    return linear_regressor, detrended
    # else:
    #     print(f'Polynomial Regression Choosed')
    #     detrended = [y_test_value[i]-polynomial_trend[i] for i in range(0, len(df['value']))]
    #     return polynomial_regressor, detrended
   

def get_polynomial_features(X_test_value):
    pf = PolynomialFeatures(degree=2)
    Xp = pf.fit_transform(X_test_value)
    return Xp

def create_ransac_model():
    ransac = RANSACRegressor(estimator=LinearRegression(),
                        min_samples=50, max_trials=100,
                        loss='absolute_loss', random_state=42,
                        residual_threshold=10)
    return ransac