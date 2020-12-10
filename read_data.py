import pandas as pd
import numpy as np
import functions as f


def people_per_country(country=None):
    """ to be implemented """
    pass


def people_per_region(region=None):
    """ Scrape an html table from a website and get the population """

    '''
    This file could also be used but it's quicker the other way
    file_regioni = 'data/Italy/DCIS_POPRES1_10122020000336854.csv'
    data = pd.read_csv(file_regioni)
    sel = data[data['Territorio'] == region]
    '''
    
    # Get the region data from this url
    region_url = 'https://www.tuttitalia.it/regioni/popolazione/'
    region_table = pd.read_html(region_url)
    
    # 
    data = region_table[0]
    data_fix = pd.DataFrame()

    # Rename the columns with some readable name
    data.columns = ['Unnamed', 'Regione', 'Popolazione', 'Superficie', 'Densita', 'NumeroComuni', 'NumeroProvince']
    nomi = []
    pop = []
    dens = []
    n = len(data)

    # The data comes in some strangr
    for i in range(0, n-1):
        nomi.append(data['Regione'].values[i])
        pop.append(int(data['Popolazione'].values[i].replace('.', '')))
        dens.append(data['Densita'].values[i])
    
    data_fix['Regione'] = np.array(nomi)
    data_fix['Densita'] = np.array(dens)
    data_fix['Popolazione'] = np.array(pop)

    data_fix.to_csv('data/Italy/popolazione_per_regione.csv')

    return data_fix[data_fix['Regione'] == region]['Popolazione'].values[0]


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
        data_clean['confirmed_velocity'] = np.gradient(data_clean['confirmed_smooth'])[::-1] 
        data_clean['confirmed_acceleration'] = np.gradient(data_clean['confirmed_velocity'])
        data_clean['deaths_smooth'] = data_clean['deaths'][::-1].rolling(window=smooth).mean()
        data_clean['deaths_variation'] = data_clean['deaths'].pct_change()
        data_clean['deaths_velocity'] = np.gradient(data_clean['deaths_smooth'])[::-1]
        data_clean['deaths_acceleration'] = np.gradient(data_clean['deaths_velocity'])

    #data_clean = data_clean.dropna()
    #print(f'Total number of points for {region} (clean) is {len(data_clean)}')
    #print(data_clean.head(20))

    return data_clean


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
        col_velocity =  columns[i] + '_velocity' 
        col_acceleration = columns[i] + '_acceleration'

        # Ensure smoothing makes sense, compute also the gradient (variatio)
        if smooth > 1:
            data_full[col_smooth] = data_full[columns[i]].rolling(window=smooth).mean()
            data_full[col_variation] = data_full[columns[i]][::-1].pct_change()
            data_full[col_variation_smooth] = data_full[col_smooth][::-1].pct_change()
            data_full[col_velocity] = np.gradient(data_full[col_smooth])
            data_full[col_acceleration] = np.gradient(data_full[col_acceleration])

    # TODO FIXME
    # Once the data has been initialized generate a target column for LSTM or other predictive models
    #for col in data.columns[1:]:
    #col_shift = col + '_target'
    #    f.shift_column(data=data_full, col_shift=col_shift, col_orig=col, n_days=n_days)
    
    return data_full



