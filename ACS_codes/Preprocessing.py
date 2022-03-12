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
    df = df.set_index('Datetime_index').resample('60min').mean()
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
    month_in_year = 12

    df['sin_hour'] = np.sin(2*np.pi*timestamps_hour/hours_in_day)
    df['cos_hour'] = np.cos(2*np.pi*timestamps_hour/hours_in_day)
    df['sin_day'] = np.sin(2*np.pi*timestamps_day/days_in_month)
    df['cos_day'] = np.cos(2*np.pi*timestamps_day/days_in_month)
    df['sin_month'] = np.sin(2*np.pi*timestamps_month/month_in_year)
    df['cos_month'] = np.cos(2*np.pi*timestamps_month/month_in_year)

    return df

train_dataset = process_data('Data/Train_raw.csv')
test_dataset = process_data('Data/Test_raw.csv')

train_dataset.to_csv(r'Data/train_dataset.csv', index=False)
test_dataset.to_csv(r'Data/test_dataset.csv', index=False)

