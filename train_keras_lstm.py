# ClearML - Keras with Tensorboard example code, automatic logging model and Tensorboard outputs
#
# Train a simple deep NN on the MNIST dataset.
# Then store a model to be served by clearml-serving
import argparse
import os
import tempfile

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
import pandas as pd
import matplotlib.pyplot as plt
import dateutil
import preprocessing
from util import plotting
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

from clearml import Task

def main():
    train_data = pd.read_csv('dataset01-disk-filesystem-incomplete_train - Copy.csv', parse_dates=['date'], names=['date', 'value'])
    test_data = pd.read_csv('dataset01-disk-filesystem-incomplete - Copy.csv', parse_dates=['date'], names=['date', 'value'])

    # drop duplicates based on date and keep last duplicate
    # train_data = train_data.drop_duplicates(subset='date', keep="last")

    train_data[['original']] = train_data[['value']]
    train_data = train_data.set_index('date')
    test_data[['original']] = test_data[['value']]
    test_data = test_data.set_index('date')

    print(f'Train shape: {train_data.shape}')
    plt.rcParams["figure.figsize"] = [18.00, 6.50]
    plt.plot(train_data)

    # seasonality = 'hourly'
    print(tf.__version__)
    regressor, train_data['value'] = preprocessing.detrend_regression(train_data)
    seasonality = preprocessing.get_seasonality(train_data)
    if seasonality is not None:
        print(f'seasonality not none: {seasonality}')
        train_data, z_score = preprocessing.z_score(train_data, seasonality)
    # train_data, z_score = preprocessing.z_score(train_data, 'hourly')
    # train_data[['value']] = preprocessing.scale(train_data[['value']])

    y_test_value = test_data['value'].values
    X_test_value = [i for i in range(0, len(test_data['value']))]
    X_test_value = np.reshape(X_test_value, (len(X_test_value), 1))
    # X_test_poly = preprocessing.get_polynomial_features(X_test_value)
    trendp = regressor.predict(X_test_value)
    detrended = [y_test_value[i]-trendp[i] for i in range(0, len(test_data['value']))]
    test_data['value'] = detrended
    if seasonality is not None:
        test_data = preprocessing.z_score(test_data, seasonality, z_score)
    # test_data[['value']] = preprocessing.scale(test_data[['value']], is_test=True)

    TIME_STEPS = 60
    x_train, y_train = preprocessing.create_sequences(train_data[['value']], train_data['value'], TIME_STEPS)
    print(f'Training shape: {x_train.shape}')
    x_test, y_test = preprocessing.create_sequences(test_data[['value']], test_data['value'], TIME_STEPS)
    print(f'Testing shape: {x_test.shape}')

    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, TimeDistributed, Dropout, GlobalMaxPool1D

    # model = Sequential()
    # model.add(Conv1D(24, 1, activation="relu", input_shape=x_train.shape[1:]))
    # model.add(Conv1D(24, 1, activation="relu"))
    # model.add(Dense(16))
    # model.add(Dense(1, activation = 'sigmoid'))
    # model.compile(loss = 'mae', 
    #             optimizer = "adam",               
    #             metrics = ['accuracy'])
    # model.summary()

    model = Sequential()
    model.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(x_train.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(x_train.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name='serving examples', task_name='test training lstm model', output_uri=True)

    output_folder = os.path.join(tempfile.gettempdir(), 'keras_example_new_temp_now')

    # load previous model, if it is there
    # noinspection PyBroadException
    try:
        model.load_weights(os.path.join(output_folder, 'weight.1.hdf5'))
    except Exception:
        pass

    from keras import callbacks
    model.fit(x_train, y_train, batch_size=16, epochs=50, verbose=1, validation_split=0.1,
                        callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')], shuffle=False)

    x_train_pred = model.predict(x_train, verbose=1)
    train_mse_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
    # train_mse_loss = tf.keras.losses.mae(x_train_pred, x_train).numpy()
    # train_mse_loss = np.concatenate(train_mse_loss, axis=0 )
    # threshold = np.mean(train_mse_loss) + np.std(train_mse_loss)

    temp_tres = np.max(train_mse_loss)
    threshold = temp_tres
    print(threshold)
    
    x_test_pred = model.predict(x_test, verbose=0)
    test_mse_loss = np.mean(np.abs(x_test_pred-x_test), axis=1)

    test_score_df = pd.DataFrame(test_data[TIME_STEPS:])
    test_score_df['loss'] = test_mse_loss
    test_score_df['threshold'] = threshold
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df['Close'] = test_data[TIME_STEPS:]['value']
    anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
    
    plotting.plot_clustering_two_ds(test_score_df, anomalies, 'original')
    plotting.plot_clustering_two_ds(test_score_df, anomalies, 'loss')

    task.upload_artifact('seasonality', seasonality)
    if seasonality is not None:
        task.upload_artifact('z_score', z_score)
    task.upload_artifact('time_steps', TIME_STEPS)
    task.upload_artifact('threshold', threshold)
    task.upload_artifact('trend_step', len(train_data))
    task.upload_artifact('regressor', regressor)
    # task.register_artifact('trend_step_v2', len(train_data))
    # upload scaler

    # store the model in a format that can be served
    model.save('serving_model', include_optimizer=False)

if __name__ == '__main__':
    main()
