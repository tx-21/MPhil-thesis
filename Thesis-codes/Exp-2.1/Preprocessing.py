import pandas as pd
import time
import numpy as np
import datetime
from icecream import ic
import matplotlib.pyplot as plt

# encoding the timestamp data cyclically. See Medium Article.
def process_data(source):

    df = pd.read_csv(source)
    df['Datetime_index'] = pd.to_datetime(df['Datetime'],format = '%Y-%m-%d %H:%M')
    # df = df.set_index('Datetime_index').resample('60min').mean()
    df = df.set_index('Datetime_index')
    df['Datetime'] = df.index
    df.reset_index(drop=True, inplace=True)

    timestamps = df['Datetime']
    timestamps_hour = pd.to_datetime(timestamps,format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.hour)
    timestamps_day = pd.to_datetime(timestamps,format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.day)
    timestamps_month = pd.to_datetime(timestamps,format='%Y-%m-%d %H:%M:%S').apply(lambda x: x.month)

    #when designing on my own position encoder, i need to come up a way to position day=1 and day=31
    #if the arr is divided by 31, then in the month with only 30 days will have a missing position
    hours_in_day = 24
    days_in_month = 30 #in the original dataset, in single sensor, the entire day time doesn't cross the whole month (which day=1 and day=31 won't have a overlapping issues on the position)
    # month_in_year = 12

    df['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
    df['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
    df['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
    df['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
    # df['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
    # df['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)

    return df


for i in range(1,9):
    problem=i
    if problem ==1:
        train_dataset = process_data("data/train/obs/train.csv")
        train_dataset.to_csv(r'data/train/obs/train_dataset.csv', index=False)
    if problem ==2:
        train_dataset = process_data("data/train/sg5/train.csv")
        train_dataset.to_csv(r'data/train/sg5/train_dataset.csv', index=False)
    if problem ==3:
        train_dataset = process_data("data/train/sg7/train.csv")
        train_dataset.to_csv(r'data/train/sg7/train_dataset.csv', index=False)
    if problem ==4:
        train_dataset = process_data("data/train/sg9/train.csv")
        train_dataset.to_csv(r'data/train/sg9/train_dataset.csv', index=False)
    if problem ==5:
        train_dataset = process_data("data/train/ew2/train.csv")
        train_dataset.to_csv(r'data/train/ew2/train_dataset.csv', index=False)
    if problem ==6:
        train_dataset = process_data("data/train/ew3/train.csv")
        train_dataset.to_csv(r'data/train/ew3/train_dataset.csv', index=False)
    if problem ==7:
        train_dataset = process_data("data/train/ew4/train.csv")
        train_dataset.to_csv(r'data/train/ew4/train_dataset.csv', index=False)
    if problem ==8:
        train_dataset = process_data("data/train/or/train.csv")
        train_dataset.to_csv(r'data/train/or/train_dataset.csv', index=False)

test_dataset = process_data('data/test/test.csv')
test_dataset.to_csv(r'data/test/test_dataset.csv', index=False)
