from urllib.request import Request, urlopen     # Import some relevant libraries just for this task
import pandas as pd
import numpy as np
import functions as f
import os 
 

# General url settings for Italian regions
global regions_path; regions_path = 'data/Italy/popolazione_per_region.csv'
global regions_url; regions_url = 'https://www.tuttitalia.it/regioni/popolazione/'

# US States population
global states_path; states_path = 'data/World/population_per_us_state.csv'
global states_url; 
#states_url='https://simple.wikipedia.org/wiki/List_of_U.S._states_by_population'
states_url = 'https://www.infoplease.com/us/states/state-population-by-rank'

# Set reference urls for global csv file
global countries_path; countries_path = 'data/World/people_per_country.csv'
global countries_url; countries_url = 'https://www.worldometers.info/world-population/population-by-country/'

# Other useful CSV files
global mobility_url; mobility_url = 'https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv'
global countries_csv_file; country_csv_file = 'data/World/OxCGRT_latest.csv'
global mobility_csv_full; mobility_csv_full='data/World/Global_Mobility_Report.csv'
global mobility_csv_reduced; mobility_csv_reduced='data/World/Global_Mobility_Report_Reduced.csv'
global mobility_csv_italy; mobility_csv_italy='data/Italy/Global_Mobility_Report_Italy.csv'


def init_data():
    """ Check if tables are there else scrap them from the web """

    # Check if this file exists
    if os.path.isfile(countries_path):
        print(f'Reading population data from {countries_path}')
        data = pd.read_csv(countries_path)    
    
    # Otherwise get the region data from an url
    else:

        print(f'Reading population data from {countries_url}')

        # We need to do this otherwise the request will be blocked
        req = Request(countries_url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        
        countries_table = pd.read_html(webpage)
    
        # Convert the scraped URL into a pandas dataframe
        data = countries_table[0]

        # Export file to CSV
        data.to_csv(countries_path)

    # Check if this file exists
    if os.path.isfile(regions_path):
        print(f'Reading population data from {regions_path}...')
        data_fix = pd.read_csv(regions_path)    
    
    # Otherwise get the region data from an url
    else:
        print(f'Reading population data from {regions_url}')
        region_table = pd.read_html(regions_url)
    
        # Convert the scraped URL into a pandas dataframe
        data = region_table[0]
        data_fix = pd.DataFrame()

        # Rename the columns with some readable name
        data.columns = ['Unnamed', 'Regione', 'Popolazione', 'Superficie', 'Densita', 'NumeroComuni', 'NumeroProvince']
        nomi = []
        pop = []
        dens = []
        n = len(data)

        # Fix the data format, it comes in some kind of list format so it needs to be adapted
        for i in range(0, n-1):
            nomi.append(data['Regione'].values[i])
            pop.append(int(data['Popolazione'].values[i].replace('.', '')))
            dens.append(data['Densita'].values[i])
    
        # Now that the data has been fixed properly
        data_fix['Regione'] = np.array(nomi)
        data_fix['Densita'] = np.array(dens)
        data_fix['Popolazione'] = np.array(pop)

        # Export file to CSV
        data_fix.to_csv(regions_path)

    # Check mobility data and export only regions of interest
    col_region = 'sub_region_1'
    col_country = 'country_region'

    if os.path.isfile(mobility_csv_reduced):
        pass
    else:
        print('Compressing mobility data...')

        # If the reduced mobility hasn't been produced yet, then download it
        if os.path.isfile(mobility_csv_full) == False:
            print(f'{mobility_csv_full} not found, downloading...')
            bash_command = "wget https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv data/World/"
            os.system(bash_command)
        
        # Now read the file and remove the details, export entire country data only
        mobility = pd.read_csv(mobility_csv_full)
        mobility = mobility[mobility[col_region].isnull()]
        mobility.to_csv(mobility_csv_reduced)

    # Same thing for the Italian data ordered by region
    if os.path.isfile(mobility_csv_italy) == False:
    
        print(f'Exporting Italian mobility data to {mobility_csv_italy}')
        mobility = pd.read_csv(mobility_csv_full)
        mobility = mobility[mobility[col_country] == 'Italy']
        mobility.to_csv(mobility_csv_italy)

    if os.path.isfile(states_path) == False:
        print(f'Exporting US population data to {states_url}')
 
        # We need to do this otherwise the request will be blocked
        req = Request(states_url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()
        
        states_table = pd.read_html(webpage)

        # Convert the scraped URL into a pandas dataframe
        data = states_table[0]

        # Export file to CSV
        data.to_csv(states_path)


    print('Done.')

    return None


def mobility(countries=None, do_type='countries', day_init=0, day_end=None):
    """ Returns mobility daya for a list of countries """

    print(f'Reading compressed mobility data from {mobility_csv_reduced}')

    regions_col_1 = 'sub_region_1'
    regions_col_2 = 'sub_region_2'
    country_col = 'country_region'
    recreation_col = 'retail_and_recreation_percent_change_from_baseline'

    if do_type == 'regions':
        data = pd.read_csv(mobility_csv_italy)
        data = data[data[regions_col_2].isnull()]

    elif do_type == 'countries':
        data = pd.read_csv(mobility_csv_reduced)

    elif do_type == 'states':   # TODO
        pass

    mobs = []
    meds = []

    for country in countries:
        this_mob = data[data[country_col] == country][recreation_col].values[day_init:day_end]
        mobs.append(np.array(this_mob, dtype=float))
        meds.append(np.median(this_mob))

    print('Done.')

    return mobs, meds


def people_per_country(countries=None):
    """ Read world population data from a table which can be scraped from the web """
    
    # Column names
    col_country = 'Country (or dependency)'  
    col_population = 'Population (2020)'

    # First let's check we're receiving a list, otherwise convert to list
    if isinstance(countries,list):
        pass
    else:
        countries = [countries]

    # Read from the local path of country files
    data = pd.read_csv(countries_path)

    # Now search for and append to this list
    populations = []

    for country in countries:
        try:
            population = data[data[col_country] == country][col_population].values
            populations.append(population)
        except ValueError:
            print(f'Country: {country} could not be found in database')

    return populations


def people_per_region(regions=None):
    """ Scrape an html table from a website and get the population """
 
    # First let's check we're dealing with a list, otherwise convert
    if isinstance(regions, list):
        pass
    else:
        regions = [regions]

    # Read from the local path
    data = pd.read_csv(regions_path)

    # Now search for and append all the values to this list
    populations = []

    for region in regions:
        if region in data['Regione'].values:
            population = data[data['Regione'] == region]['Popolazione'].values[0]
        else:
            population = -1

        populations.append(population)

    return populations


def people_per_state(states=None):
    """ Read a table with states information and population """
    
    if isinstance(states, list):
        pass
    else:
        regions = [regions]

    data = pd.read_csv(states_path)
    
    populations = []

    for state in states:
        if state in data['State'].values:
            population = data[data['State'] == state]['Population'].values[0]
        else:
            population = -1

        populations.append(population)

    return populations


def country_data(countries=None, verbose=False, day_start=0):
    """ This reads and formats the full Oxford dataset """

    data = pd.read_csv(country_csv_file)
    populations = people_per_region(regions=countries)

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
    col_stringency = cols[41]
    col_response = cols[44]
    col_contain = cols[45]
    col_deaths = cols[38]

    masks = []
    responses = []
    stringencies = []

    print(col_response, col_stringency, col_contain)

    for i, country in enumerate(countries):
        pop = populations[i] / 1.e+6
        sel_data = data[data[col_country] == country][day_start:]

        # Enforce this check or things will be screwed
        sel_data = sel_data[sel_data['RegionName'].isnull()]
        
        col_i = col_mask
        col_ii = col_response
        col_iii = col_stringency

        n_use = len(sel_data[col_i])
        print(f'Using {n_use} points for {country}')

        masks.append(sel_data[col_i].mean())
        responses.append(sel_data[col_ii].mean())
        stringencies.append(sel_data[col_iii].mean())

    return masks, responses, stringencies


def smooth_data(data=None, smooth=3, invert=False, inverse_smooth=True):
    """ Smooth the columns over a rolling average of n=smooth days and remove outliers """

    def remove_outliers(x):
        """ This function can be made more sophisticated """

        if x < 0.0: 
            x = 0.0

        return x

    
    def inverse_smooth(x):
        """ Take the average over n_smooth days before the last one """

        x_new = np.zeros(len(x))

        for i, elem in enumerate(x):
            x_new[i] = np.mean(x[i:i+smooth])

        return x_new


    if smooth > 1:
        print(f'Smoothing data over {smooth} days')
        columns = ['confirmed', 'deaths']
    else:
        print('smooth_data() will not work, smoothing days={smooth}')

        return data

    # Loop over the columns we want to smooth over
    for col in columns:

        data[col] = data[col].apply(remove_outliers)

        # Smoothed data goes to a new column
        col_smooth = col + '_smooth'
        col_variation = col + '_variation'
        col_variation_smooth = col+ '_variation_smooth'
        col_velocity =  col + '_velocity' 
        col_acceleration = col + '_acceleration'

        if inverse_smooth:
            data[col_smooth] = inverse_smooth(data[col].T.values) #(inverse_smooth)
        else:
            data[col_smooth] = data[col].rolling(window=smooth).mean()

        # TODO 
        if invert:
            data[col_variation] = data[col][::-1].pct_change()
            data[col_variation_smooth] = data[col_smooth][::-1].pct_change()
            data[col_velocity] = np.gradient(data[col_smooth])
            data[col_acceleration] = np.gradient(data[col_velocity])
        else:
            data[col_variation] = data[col].pct_change()
            data[col_variation_smooth] = data[col_smooth].pct_change()
            data[col_velocity] = -np.gradient(data[col_smooth])
            data[col_acceleration] = np.gradient(data[col_velocity])

    return data


def extract_state(state=None, smooth=1, n_days=1):
    """ This function is specific only for italian regions """

    data = pd.read_csv(country_csv_file)

    col_state = 'RegionName'
    col_cases = 'ConfirmedCases'
    col_deaths = 'ConfirmedDeaths'

    if state not in data[col_state].unique():
        print('State: ', region, ' is not a valid state. Try again.')
        exit()

    print(f'State selected: {state}')
    names_states = data[col_state].unique()
    data_clean = pd.DataFrame()
    data_state = data[data[col_state] == state]

    print(f'Total number of points for {state} is {len(data_state)}')
    data_clean['date'] = data_state['Date'][::-1]
    data_clean['confirmed'] = f.differential(cumulative=data_state['ConfirmedCases'][::-1].values)
    data_clean['deaths'] = f.differential(cumulative=data_state['ConfirmedDeaths'][::-1].values)
    #data_clean['confirmed'] = data_state['ConfirmedCases'][::-1].values
    #data_clean['deaths'] = data_state['ConfirmedDeaths'][::-1].values

    data_clean = smooth_data(data=data_clean, smooth=smooth)

    return data_clean


def extract_region(region=None, smooth=1, n_days=1):
    """ This function is specific only for italian regions """

    file_regioni = 'data/Italy/dati-regioni/dpc-covid19-ita-regioni.csv'
    data_regioni = pd.read_csv(file_regioni)
    col_region = 'denominazione_regione'

    if region not in data_regioni[col_region].unique():
        print('Region: ', region, ' is not a valid region. Try again.')
        exit()

    print(f'Region selected: {region}')
    nomi_regioni = data_regioni[col_region].unique()
    data_clean = pd.DataFrame()
    data_region = data_regioni[data_regioni[col_region] == region]

    print(f'Total number of points for {region} is {len(data_region)}')
    data_clean['date'] = data_region['data'][::-1]
    data_clean['confirmed'] = data_region['nuovi_positivi'][::-1]
    data_clean['deaths'] = f.differential(cumulative=data_region['deceduti'][::-1].values)

    data_clean = smooth_data(data=data_clean, smooth=smooth)

    return data_clean


def extract_country(n_days=1, country=None, smooth=7):
    """ Take a single country from the fulldataset and extract the relevant time series """

    covid_path = 'data/World/csse_covid_19_data/csse_covid_19_time_series/'
    deaths_file = 'time_series_covid19_deaths_global.csv'
    confirmed_file = 'time_series_covid19_confirmed_global.csv'
    recovered_file = 'time_series_covid19_recovered_global.csv'
    col_name = 'Country/Region'
    province_name = 'Province/State'
    
    confirmed = pd.read_csv(covid_path + confirmed_file)
    deaths = pd.read_csv(covid_path + deaths_file)

    columns = ['confirmed', 'deaths']
    data_full = pd.DataFrame()

    # Loop over the two data types, extract also smoothed values and daily variation rates
    for i, data in enumerate([confirmed, deaths]):

        dates = data.columns[4:]
        row = data[(data[col_name] == country) & (data[province_name].isnull())].T.iloc[4:].values
        row = np.array(row)

        if len(row) == 0:
            print(f'{country} has zero points')
            
            return data_full
        else:
            row = np.reshape(row, len(row))
    
        # Date & differential number of cases
        data_full['date'] = dates
        data_full[columns[i]] = f.differential(cumulative=row[::-1])

    data_full = smooth_data(data=data_full, smooth=smooth)

    # TODO FIXME
    # Once the data has been initialized generate a target column for LSTM or other predictive models
    #for col in data.columns[1:]:
    #col_shift = col + '_target'
    #    f.shift_column(data=data_full, col_shift=col_shift, col_orig=col, n_days=n_days)
    
    return data_full


if __name__ == "__main__":
    """ Test the routines """

    countries = ['Italy', 'France', 'Sweden']
    states = ['South Dakota', 'Florida', 'North Dakota', 'California']
    init_data()

    pops = people_per_state(states=states)

    for i, p in enumerate(pops):
        print(f'{states[i]} {p}')
        state = extract_state(state=states[i])

    '''
    mobs = mobility(countries=countries)

    for i in range(0, len(countries)):
        med = np.median(mobs[i])
        print(f'{countries[i]} MobData: {med}')
    '''

    #print(people_per_region(regioni='Lazio'))
    #print(people_per_country(countries='Italy'))

