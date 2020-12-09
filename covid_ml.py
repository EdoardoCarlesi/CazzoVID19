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


def extract_region(region=None, smooth=1):
    """ This function is specific only for italian regions """

    file_regioni = 'data/Italy/dati-regioni/dpc-covid19-ita-regioni.csv'
    data_regioni = pd.read_csv(file_regioni)

    #print(data_regioni.columns)

    if region not in data_regioni['denominazione_regione'].unique():
        print('Region: ', region, ' is not a valid region. Try again.')
        exit()

    print('Region selected: ', region)
    #print(data_regioni.head())
    #print(data_regioni[data_regioni['denominazione_regione'] == 'Abruzzo']['nuovi_positivi'])
    nomi_regioni = data_regioni['denominazione_regione'].unique()
    data_clean = pd.DataFrame()
    data_regione = data_regioni[data_regioni['denominazione_regione'] == region]
    print(f'Total number of points for {region} is {len(data_regione)}')
    #print(data_regione.head())

    data_clean['date'] = data_regione['data'][::-1]
    data_clean['confirmed'] = data_regione['nuovi_positivi'][::-1]
    data_clean['deaths'] = f.differential(cumulative=data_regione['deceduti'][::-1].values)
    #print(data_clean['deaths'])


    if smooth > 1:
        data_clean['confirmed_smooth'] = data_clean['confirmed'][::-1].rolling(window=smooth).mean()
        data_clean['confirmed_variation_smooth'] = data_clean['confirmed_smooth'][::-1].pct_change()
        data_clean['confirmed_variation'] = data_clean['confirmed'][::-1].pct_change()
        data_clean['deaths_smooth'] = data_clean['deaths'][::-1].rolling(window=smooth).mean()
        data_clean['deaths_variation'] = data_clean['deaths'].pct_change()
        data_clean['deaths_variation_smooth'] = data_clean['deaths_smooth'].pct_change()

    #data_clean = data_clean.dropna()
    #print(f'Total number of points for {region} (clean) is {len(data_clean)}')
    #print(data_clean.head(20))

    return data_clean


def country_data(countries=None, populations=None, verbose=False):
    """ This reads and formats the full Oxford dataset """

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


def extract_country(n_days=1, country=None, smooth=7):
    """ Take a single country from the fulldataset and extract the relevant time series """

    covid_path = '/home/edoardo/devel/CoronaVirus/data/World/csse_covid_19_data/csse_covid_19_time_series/'
    deaths_file = 'time_series_covid19_deaths_global.csv'
    confirmed_file = 'time_series_covid19_confirmed_global.csv'
    recovered_file = 'time_series_covid19_recovered_global.csv'
    col_name = 'Country/Region'
    
    confirmed = pd.read_csv(covid_path + confirmed_file)
    deaths = pd.read_csv(covid_path + deaths_file)

    columns = ['confirmed', 'deaths']
    data_full = pd.DataFrame()

    # Loop over the two data types, extract also smoothed values and daily variation rates
    for i, data in enumerate([confirmed, deaths]):

        dates = data.columns[4:]
        row = data[data[col_name] == country].T.iloc[4:].values
        row = np.array(row)
        row = np.reshape(row, len(row))
    
        # Date & differential number of cases
        data_full['date'] = dates
        data_full[columns[i]] = f.differential(cumulative=row[::-1])

        # Smooth the data to another column
        col_smooth = columns[i] + '_smooth'
        col_variation = columns[i] + '_variation'
        col_variation_smooth = columns[i] + '_variation_smooth'

        # Ensure smoothing makes sense, compute also the gradient (variatio)
        if smooth > 1:
            data_full[col_smooth] = data_full[columns[i]].rolling(window=smooth).mean()
            data_full[col_variation] = data_full[columns[i]][::-1].pct_change()
            data_full[col_variation_smooth] = data_full[col_smooth][::-1].pct_change()

    # TODO FIXME
    # Once the data has been initialized generate a target column for LSTM or other predictive models
    #for col in data.columns[1:]:
    #col_shift = col + '_target'
    #    f.shift_column(data=data_full, col_shift=col_shift, col_orig=col, n_days=n_days)
    
    return data_full


def forward_prediction(days_fwd=1, model=None, start=None):
    """ If we have a model, let's see what the predictions are for the next days """

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
 

def montecarlo_fit(function=None, params=None, intervals=None, x=None, y=None, n=10000):
    """ Fit a generic function with a broad MC search of the parameter space """

    print(f'Fitting {function.__name__} with MC method...')
    n_p = len(intervals)
    params_mc = np.zeros((n, n_p))
    err_mc = np.zeros(n)

    intervals[2][0] = np.log(intervals[2][0])
    intervals[2][1] = np.log(intervals[2][1])

    for i in range(0, n_p):
        params_mc[:, i] = np.random.uniform(low=intervals[i][0], high=intervals[i][1], size=n)

    params_mc[:, i] = np.exp(params_mc[:, i])

    for j in range(0, n):
        this_y = function(x, *params_mc[j,:]) #, params_mc[j,1], params_mc[j,2])
        err_mc[j] = np.std(abs(this_y - y))

    params = params_mc[np.argmin(err_mc)]

    print(f'ErrMin: {min(err_mc)}, ErrMax: {max(err_mc)} with params={params}')
    return params


def fit_gompertz(x=None, y=None, n=3000, montecarlo=False):
    """ Fit a Gompertz function  with three parameters """

    def gompertz_scipy(t, a, b, c):
        """ Gompertz curve declared in a scipy compliant form """
    
        return f.gompertz(t=t, a=a, b=b, c=c, derive=True)

    intA = [1.0, 100.0]
    intB = [1.0, 500.0]
    intC = [0.00001, 0.1]
    params = [1.0, 1.0, 1.0]

    if montecarlo:
        intervals = [intA, intB, intC]
        params = montecarlo_fit(function=gompertz_scipy, intervals=intervals, x=x, y=y, n=n)

    try:
        popt, pcov = curve_fit(gompertz_scipy, xdata=x, ydata=y, p0=params)    
        print(f'Best fit={popt}')
    except:
        print('Best fit parameters not found, using MC instead...')
        popt = params

    g_mc = f.gompertz(t=x, a=params[0], b=params[1], c=params[2], derive=True)
    g_fit = f.gompertz(t=x, a=popt[0], b=popt[1], c=popt[2], derive=True)

    return g_mc, g_fit


def compare_curves(do_countries=False, do_regions=True, countries=[]):
    """ Fit data from various countries/regions to Gompertz curve """

    # How many days forward for the target and on how many days should we smooth (rolling average)
    n_days = 1
    n_smooth = 12

    # What kind of analysis?
    montecarlo = False
    do_gompertz = False
    shift_peak = True
    do_bin = False

    if do_countries:
        #countries = ['Sweden', 'Italy']
        #countries = ['Italy', 'Belgium', 'Serbia']; populations = [62e+6, 11.5e+6, 7.5e+6]
        countries = ['Italy', 'Czechia', 'Slovakia', 'Germany', 'Belgium', 'Sweden'] 
        #countries = ['Italy', 'Belgium', 'Norway', 'Finland', 'Slovakia', 'Germany'] 
        #countries = ['Sweden', 'Italy', 'Belgium', 'Serbia']
 
    elif do_regions:
        countries = ['Lombardia', 'Veneto', 'Lazio', 'Abruzzo'] 
        #countries = ['Abruzzo', 'Lombardia', 'Lazio', 'Veneto', 'Campania']
        #countries = ['Abruzzo', 'Lombardia', 'Lazio', 'Sicilia', 'Veneto', 'Campania']

    #columns = ['confirmed_smooth', 'deaths_smooth']
    #columns = ['confirmed_variation_smooth', 'confirmed_smooth']
    #columns = ['confirmed_smooth', 'confirmed']
    #columns = ['confirmed_smooth', 'deaths_smooth']
    #columns = ['deaths_smooth']
    #columns = ['deaths_variation', 'deaths_smooth']
    #columns = ['confirmed_variation', 'confirmed_smooth']
    columns = ['confirmed_smooth']
    #columns = ['deaths_smooth']
    #columns = ['confirmed_variation_smooth']
    #columns = ['confirmed_variation_smooth']
    #columns = ['confirmed_variation_smooth', 'deaths_variation_smooth']

    for i, country in enumerate(countries):

        if do_countries:
            data = extract_country(n_days=n_days, smooth=n_smooth, country=country)     
            data = data.dropna()
        elif do_regions:
            data = extract_region(region=country, smooth=n_smooth)
            #data = data.dropna()

        t_min = 220
        t_max = 320 - n_smooth
        ts = np.arange(0, t_max-t_min)
        #pop0 = populations[i]

        title = ' '.join(countries) + ' ' + str(n_smooth) + ' day average'
        if shift_peak:
            title += ' peak centered'

        plt.title(title)

        print(data.head())

        for select in columns:
            #values = data[select].values[0:t_max-t_min][::-1]
            values = data[select].values[0:t_max-t_min]
            values0 = np.max(values[np.logical_not(np.isnan(values))])

            if do_bin:
                bin_df = f.bin_mean(values)
                values = bin_df['mean']
                ts = bin_df['t']

            print(f'N Values: {len(values)}, max:{values0}')
    
            if values0 > 1.0:
                values /= values0
                t_value = np.where(values == 1.0)
                values *= values0

            #values /= pop0
            #values *= 100.0

            data_label = 'data_'+select+'_'+country
            fit_label = 'fit_'+select+'_'+country
            mc_label = 'mc_'+select+'_'+country

            if shift_peak:
                t_shift = t_value[0][0]
                
                ts[:] -= t_shift

            plt.plot(ts, values, label=data_label)

            if do_gompertz:
                g_mc, g_fit = fit_gompertz(x=ts, y=values, montecarlo=True)
                plt.plot(ts, g_fit, label=fit_label)

                if montecarlo:
                    plt.plot(ts, g_mc, label=mc_label) 

    plt.legend()
    plt.show() 
    #block=False)
    #plt.pause(7)
    #plt.close()


if __name__ == "__main__":

    """ MAIN PROGRAM """

    #countries = country_data()

    #compare_curves(do_countries=True, do_regions=False)
    compare_curves()





