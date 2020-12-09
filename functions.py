import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import matplotlib.pyplot as plt

import time
import functools
import pandas as pd
import numpy as np

#import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from scipy import stats
from copy import copy


def people_per_region():

    ppr = dict()

    ppr['Lombardia'] = 10.104e+6


    return ppr


def differential(cumulative=None):
    """
        Given a cumulative distribution get the differential one
    """

    n_diff = len(cumulative)
    differential = np.zeros(n_diff)
    differential[n_diff-1] = cumulative[n_diff-1]

    for i in range(0, n_diff-1):
        differential[i] = cumulative[i] - cumulative[i+1]

    return differential


def variation(xdata=None):
    """
        Given a cumulative distribution get the differential one
    """

    n_var = len(xdata)
    variation = np.zeros(n_var)

    variation[n_var-1] = 0.0 #xdata[n_var-1]
    variation[n_var-2] = 0.0 #xdata[n_var-1]

    for i in range(0, n_var-1):

        if xdata[i+1] != 0:
            variation[i] = (xdata[i]-xdata[i+1]) / xdata[i+1]
        else:
            variation[i] = 0.0

    return variation


def bin_mean(y=None, dx=7):
    """
        Average value over a dx interval for a set of y values
    """

    bin_df = pd.DataFrame()
    nbins = int(len(y) / dx)

    t_bin = np.zeros(nbins+1)
    y_bin = np.zeros(nbins+1)
    y_max = np.zeros(nbins+1)
    y_min = np.zeros(nbins+1)

    x = 0

    for i in range(0, nbins):
        y_bin[i] = np.mean(y[i*dx:(i+1)*dx])  
        y_max[i] = np.max(y[i*dx:(i+1)*dx])  
        y_min[i] = np.min(y[i*dx:(i+1)*dx])  
        t_bin[i] = i

    t_bin[nbins] = nbins
    y_bin[nbins] = np.mean(y[(nbins-1)*dx:])  
    y_max[nbins] = np.max(y[(nbins-1)*dx:])  
    y_min[nbins] = np.min(y[(nbins-1)*dx:])  

    bin_df['t'] = t_bin
    bin_df['mean'] = y_bin
    bin_df['max'] = y_max
    bin_df['min'] = y_min

    print(bin_df.head())

    return bin_df


def gompertz(t=None, a=None, b=None, c=None, derive=True, verbose=False):
    """
        Gompertz function
        This is a cumulative function, if derive == True then compute the derivative (assuming time unit = 1)
    """
    
    if verbose:
        print(f'Using Gompertz function with a={a}, b={b}, c={c}')

    f_t = a * np.exp(-b * np.exp( -c * t))
    
    if derive:
        n = len(f_t)
        f_t_prime = np.zeros(n)

        for i in range(1, n):
            f_t_prime[i] = f_t[i] - f_t[i-1]

        return f_t_prime

    else:
        return f_t


def prepare_data(normalize=True, data=None, split_fac=0.7, LSTM=False, date_col='date', n_days=1):
    """
        Normalize and split the data into train and test set
        Ordering here is important so we cannot use sklearn directly
    """
   
    if normalize:
        # Rescale all data to the (0,1) interval
        sc = MinMaxScaler(feature_range = (0, 1))
        data = sc.fit_transform(data.drop(columns = [date_col]))

    # If using Keras for LSTM we need another kind of format for X and y
    if LSTM == True:
        X = []
        y = []

        for i in range(1, len(data)):
            X.append(data[i-1:i, 0])
            y.append(data[i, 0])

        X = np.asarray(X)
        y = np.asarray(y)

    else:
        X = data[:, :2]
        y = data[:, 2:]

    split = int(split_fac * len(X))

    X_train = X[:split];     y_train = y[:split]
    X_test = X[split:];     y_test = y[split:]

    return X_train, y_train, X_test, y_test


def show_plot(data=None, title=None):
    """
        Simple lineplot
    """

    plt.figure(figsize = (15, 7))
    plt.plot(data, linewidth = 3)
    plt.title(title)
    plt.grid()

    plt.show(block=False)
    plt.pause(3)
    plt.close()


def shift_column(data=None, n_days=1, col_shift='Target', col_orig=None):
    """
        We shift all streams of n_days
    """
    
    data[col_shift] = data[[col_orig]].shift(-n_days)

    return data


def time_total(function):
    """ 
        Wrapper function to decorate other functions and get their running time 
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()

        print(f'Function {function.__name__}() took {t1-t0} seconds to execute.')
        
        return result 

    return wrapper


def normalize(data=None):
    """
        Normalize all data to 1.0 at their starting values
    """

    x = data.copy()

    for i in x.columns[1:]:
        x[i] = x[i]/x[i][0]

    return x


def interactive_plot(data=None, title=None, mode="lines", x_col='date'):
    """
        This is an interactive plot generator for the browser, using plotly
    """

    if mode == "lines":
        fig = px.line(title = title)
    
        for col in data.columns[1:]:
            fig.add_trace(go.Scatter(x = data[x_col], y = data[col], mode=mode, name = col))

    # In histogram mode we assume data is already 1-dimensional
    elif mode == "histogram":
        fig = px.histogram(data, title = title)

    elif mode == "scatter":
        pass

    fig.show()


if __name__ == "__main__":
    """
        The main is used to test functions
    """


    """
    t = np.arange(0, 100, 1)
    f, fp = gompertz(t=t, a=1.0, b=20, c=0.1)
    print(f)
    print(fp)
    plt.plot(t, fp)
    plt.show()
    """


