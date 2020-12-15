import pandas as pd
import numpy as np
import functions as f

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats
from copy import copy

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


def train_lr(data=None, n_days=1, split_fac=0.7, plot=True, lr_type='simple'):
    
    """
        Train a linear regression model
    """

    if lr_type == 'simple':
        pass
    elif lr_type == 'Ridge':
        pass


def train_lstm(data=None, col_date='date', n_days=1, split_fac=0.7, 
        plot=True, n_units=100, drop_fac=0.3, n_epochs=15, n_batch=10, split_size=0.2, model=None):
    
    from tensorflow import keras

    """
        Train a LSTM neural network for prediction
        The parameters for the NN architecture and training have to be specified:

        n_units = 100
        drop_fac = 0.2
        n_epochs = 15
        n_batch = 10
    """

    # Do the train - test splitting with normalized data
    X_train, y_train, X_test, y_test = f.prepare_data(data=data, split_fac=split_fac, LSTM=True)
    X_train = np.asarray(X_train)

    split = len(y_train)
    print(f'Train test split, train = {split}, Shape of X_Train/Test: {X_train.shape}')

    # Reshape the arrays for keras
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print(f'Reshaping X_Train/Test: {X_train.shape}')

    if model == None:
        # Build the network
        inputs = keras.layers.Input(shape = (X_train.shape[1], X_train.shape[2]))

        x = keras.layers.LSTM(n_units,return_sequences=True)(inputs)
        x = keras.layers.Dropout(drop_fac)(x)
        x = keras.layers.LSTM(n_units)(x)
        output = keras.layers.Dense(1, activation='linear')(x)

        model = keras.models.Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        print(model.summary())
    
        history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=n_batch, validation_split=split_size)

    predictions = model.predict(X_test)
    
    pred = []
    for p in predictions:
        pred.append(p[0])

    data_new = pd.DataFrame()
    data_new[col_date] = data[col_date].iloc[split:-1]
    data_new['True'] = y_test 
    data_new['Pred'] = np.array(pred)
    title = 'LSTM for N days=' + str(n_days)
    f.interactive_plot(data=data_new, title=title)

    return model, predictions


def do_lstm():
    """ Train an LSTM model for the time series """

    #col_country = 'Japan'
    col_country = 'Italy'
    n_days = 1
    n_smooth = 10
    
    data = extract_country(n_days=n_days, smooth=n_smooth, col_country=col_country)

    print(data.head())
    
    # Prepare data for keras
    data = data.dropna()
    print(data.tail())
    print('New data size: ', len(data))

    split_fac = 0.7
    n_units = 100
    drop_fac = 0.2
    n_epochs = 25
    n_batch = 10

    model, pred = train_lstm(data=data, n_days=n_days, split_fac=split_fac, plot=True, col_date=col_date,  
            n_units=n_units, drop_fac=drop_fac, n_epochs=n_epochs, n_batch=n_batch)

    #col_country = 'Spain'
    col_country = 'Sweden'
    #col_country = 'Japan'

    data = extract_country(data=covid_data, n_days=n_days, col_name=col_name, smooth=n_smooth, 
                col_country=col_country, col_date=col_date, col_confirmed=col_confirmed)

    data = data.dropna()
    print(data.head())
    
    max_value = data[col_confirmed].max()

    days_fwd = 31
    data_new = data[:-days_fwd]
    data_test = data[-days_fwd:]
    print(data_test.head())
    model, pred = train_lstm(data=data_new, n_days=n_days, split_fac=0.0, plot=True, col_date=col_date, model=model)

    print(f'StartingValue: {start}')
    forward = forward_prediction(days_fwd=days_fwd, model=model, start=start) * max_value
    
    data_test['fwd'] = forward
    print(data_test[col_confirmed])
    plt.plot(data_test[col_confirmed])
    plt.plot(data_test['fwd'])
    #plt.plot(data_test)
    #plt.plot(forward)
    plt.show()
    
    #print(pred)
    print(forward)
 

if __name__ == "__main__":

    """ Test functions """
    pass

