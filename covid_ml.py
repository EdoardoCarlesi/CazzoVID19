import pandas as pd
import numpy as np
import functions as f

import matplotlib.pyplot as plt

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

def italian_data():
    pass


def country_data(countries=None, populations=None, verbose=False):
    """
        
    """

    csv_file = '/home/edoardo/devel/CoronaVirus/data/CountryInfo/OxCGRT_latest.csv'

    data = pd.read_csv(csv_file)

    print(data.info())
    if verbose:
        print(data.info())
        print(data.head())

    # Useful colum names to keep track of
    cols = data.columns
    col_country = cols[0]
    col_school = cols[6]
    col_work = cols[7]
    col_events = cols[10]
    col_transport = cols[18]
    col_travel = cols[20]
    col_home = cols[16]
    col_test = cols[28]
    col_trace = cols[29]
    col_mask = cols[32]
    col_stringency = cols[37]
    col_response = cols[41]
    col_contain = cols[43]

    col_deaths = cols[36]

    countries = ['Italy', 'Sweden', 'Denmark', 'Germany', 'Spain', 'France', 'Russia', 'Japan', 'Peru', 'Brazil']; 
    populations = [62.0, 10.0, 5.0, 80.0, 45.0, 60, 200, 120, 35, 300]

    for i, country in enumerate(countries):
        pop = populations[i]
        sel_data = data[data[col_country] == country]
    
        #col_use = col_mask
        #col_use = col_stringency
        col_x = col_response
        col_y = col_deaths

        n_use = len(sel_data[col_x])
    
        print(f'Using {n_use} points for {country}')

        sel_data = sel_data[[col_x, col_y]].dropna()

        this_x = [i for i in range(0, n_use)]

        #plt.plot(sel_data[col_x].mean(), sel_data[col_y].mean(), label=country)
        plt.scatter(sel_data[col_x].mean(), sel_data[col_y].mean()/pop, label=country)
    
    plt.legend()
    plt.show()


def world_data(country=None):
    
    # Read and normalize the data
    covid_path = '/home/edoardo/devel/CoronaVirus/data/World/csse_covid_19_data/csse_covid_19_time_series/'
    deaths_file = 'time_series_covid19_deaths_global.csv'
    recovered_file = 'time_series_covid19_recovered_global.csv'
    confirmed_file = 'time_series_covid19_confirmed_global.csv'
    data_confirmed = pd.read_csv(covid_path + confirmed_file)
    
    # TODO: Concatenate with deaths and recovered
    covid_data = data_confirmed


    return covid_data


def extract_country(data=None, n_days=1, col_name=None, col_country=None, col_date=None, col_confirmed=None, cumulative=True, smooth=0):
    dates = covid_data.columns[4:]
    row = covid_data[covid_data[col_name] == col_country].T.iloc[4:].values
    row = np.array(row)
    row = np.reshape(row, len(row))

    data = pd.DataFrame()
    data[col_date] = dates

    new_row = np.zeros(len(row))
    new_row[0] = row[0]

    if cumulative:
        for i in range(1, len(row)):
            new_row[i] = row[i] - row[i-1]

        data[col_confirmed] = new_row
    else:
        data[col_confirmed] = row

    if smooth > 1:
        data[col_confirmed] = data[col_confirmed].rolling(window=smooth).mean() 

    for col in data.columns[1:]:
        col_shift = col + '_target'
        f.shift_column(data=data, col_shift=col_shift, col_orig=col, n_days=n_days)
    
    return data


def forward_prediction(days_fwd=1, model=None, start=None):

    fwd = np.zeros(days_fwd)
    
    start = np.reshape(start, (1, 1, 1))
    #X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    v = model.predict(start)
    fwd[0] = v
    v = np.reshape(v, (1, 1, 1))

    for d in range(1, days_fwd):
        v = model.predict(v)
        fwd[d] = v
        v = np.reshape(v, (1, 1, 1))

    return fwd


if __name__ == "__main__":

    """ MAIN PROGRAM """

    col_name = 'Country/Region'
    #col_country = 'Japan'
    col_country = 'Italy'
    col_date = 'date'
    col_confirmed = 'confirmed'
    n_days = 1
    n_smooth = 10


    countries = country_data()



    '''
    covid_data = world_data()
    
    data = extract_country(data=covid_data, n_days=n_days, col_name=col_name, smooth=n_smooth,
                col_country=col_country, col_date=col_date, col_confirmed=col_confirmed)

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
    '''



