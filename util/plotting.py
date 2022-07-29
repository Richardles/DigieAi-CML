from turtle import color
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot(dataset, y_column, start_time = 0, end_time=None, report_task=None, series=''):
    fig, ax = plt.subplots(1,figsize=(20,10))
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(xfmt)

    plt.plot(dataset[start_time:end_time].index, dataset[start_time:end_time][y_column])

    if report_task is not None:
        report_task.logger.report_matplotlib_figure(
            title=f'Plot {y_column}',
            series=series,
            iteration=0,
            figure=plt,
            report_image=True
        )
    plt.show()

def plot_clustering_approach(dataset, y_column, labels, start_time = 0, end_time=None, report_task=None, series=''):
    fig, ax = plt.subplots(1,figsize=(20,10))
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(xfmt)

    plt.plot(dataset.index[start_time:end_time], dataset[y_column][start_time:end_time], color='b')
    plt.scatter(dataset[labels == -1].index, dataset[labels == -1][y_column], c='r', s=20)

    if report_task is not None:
        report_task.logger.report_matplotlib_figure(
            title=f'Plot {y_column}',
            series=series,
            iteration=0,
            figure=plt,
            report_image=True
        )
    plt.show()

def plot_clustering_two_ds(dataset, anomaly_dataset, y_column, start_time = 0, end_time=None, report_task=None, series=''):
    fig, ax = plt.subplots(1,figsize=(20,10))
    fig.autofmt_xdate()
    xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(xfmt)

    plt.plot(dataset[start_time:end_time].index, dataset[start_time:end_time][y_column], color='b')
    plt.scatter(anomaly_dataset.index, anomaly_dataset[y_column], c='r', s=20)

    if report_task is not None:
        report_task.logger.report_matplotlib_figure(
            title=f'Plot {y_column}',
            series=series,
            iteration=0,
            figure=plt,
            report_image=True
        )
    plt.show()

def train_val_loss(loss, val_loss):
    plt.plot(loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend()
    plt.show()

def loss_histogram(loss, xlabel):
    plt.hist(loss, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel('Number of Samples')
    plt.show()