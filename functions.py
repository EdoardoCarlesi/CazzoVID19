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


def smooth_data(data=None, avg_days=7):
    """
        Smooth the data over an interval window
    """

    pass

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


@time_total
def generate_portfolios(data=None, w=None, n_runs=None):
    """ 
        Assign random asset allocations n times 
        - data should be normalized to the starting price of each stock
        - n_runs is the number of montecarlo trials
    """

    portfolios = pd.DataFrame()
    portfolios['Date'] = data['Date']
    data_cols = data.columns[1:]
    n_cols = len(data_cols)

    for i_mc in range(0, n_runs):

        if w == None:
            w = normalized_random(n_cols)
        
        portfolio = make_portfolio(data=data, w=w)       

        key_p = 'P_' + str(i_mc)        
        portfolios[key_p] = portfolio[data_cols].apply(lambda x: sum(x), axis = 1)

    return portfolios


def sharpe_ratio(rp=None, rf=None, sigma=None):
    """
        The Sharpe ratio is defined as 
        SR = (R_p - R_f) / sigma_p
        R_p = return of the portfolio
        R_f = risk free return
        sigma_p = volatility (std dev) of my portfolio
    """

    return (rp - rf) / sigma


def beta(data=None, col_stock=None, col_market=None):
    """
        Beta is the basis of the CAPM model, it's the correlation between the market and a given stock
    """
    
    beta, alpha = np.polyfit(data[col_market], data[col_stock], 1)

    return beta, alpha


def capm(rf=None, beta=None, rm=None):
    """ 
        The expected return of a security is the risk-free rate of return, plus beta (the correlation between the stock and the market) 
        times the risk premium incentive i.e. the difference between the risk-free rate and the one of this stock.
        Rm is the market rate of return (e.g. S&P 500)
    """

    return (rf + beta * (rm - rf))


if __name__ == "__main__":
    """
        The main is used to test functions
    """
    pass


