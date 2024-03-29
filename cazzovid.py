import pandas as pd
import numpy as np
import functions as f
import read_data as rd


import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import datetime

from scipy.optimize import curve_fit
from scipy import stats
from copy import copy
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression


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


def correlate_shift(xdata_all=None, ydata_all=None, window=7, shift_min=7, shift_max=30):
    """ Check for the correlation between a set of variables on the x axis (e.g. policy stringency) and y (e.g. deaths) """

    npts_corr = shift_max - shift_min
    correlation = np.zeros(npts_corr)
    
    # Loop on the shifted date - the x axis is taken from 0 to shift and the y axis from shift to the end
    for shift in range(shift_min, shift_max):

        # xdata_all and ydata_all are arrays of arrays containing all the countries
        for xdata, ydata in zip(xdata_all, ydata_all):
            x_shift = xdata[:-shift]
            y_shift = xdata[shift:]
            npts_days = len(x_shift)

            # Now we will gather the data on a window basis
            x_block = np.zeros(npts_days-window)
            y_block = np.zeros(npts_days-window)

            # At each day take a chunk of size "window" and take the mean value
            for day in range(0, npts_days-window):
                x_block[day] = x_shift[(day):(day+window)]
                y_block[day] = y_shift[(day):(day+window)]

#scipy.stats.pearsonr()

    return correlation


def compare_curves(countries=None, normalize=True, columns=None, n_smooth=7, t_max=320, t_min=250, n_days=1, show=True, do_type='countries', invert=False):
    """ Fit data from various countries/regions to Gompertz curve """

    # Median and total values for the indicator which will be returned
    medians = []
    totals = []

    # What kind of analysis?
    montecarlo = False
    do_gompertz = False
    shift_peak = False
    do_bin = False
    
    # Initialize the population values
    if do_type == 'countries':
        populations = rd.people_per_country(countries=countries) 
    elif do_type ==  'regions':
        populations = rd.people_per_region(regions=countries)
    elif do_type == 'states':
        populations = rd.people_per_state(states=countries)

    # Loop on countries we want to check
    for i, country in enumerate(countries):

        if do_type == 'countries':
            data = rd.extract_country(n_days=n_days, smooth=n_smooth, country=country)
            normalization = 1.e+6
        elif do_type == 'regions':
            data = rd.extract_region(region=country, smooth=n_smooth)
            normalization = 1.e+6
        elif do_type == 'states':
            data = rd.extract_state(state=country, smooth=n_smooth)
            normalization = 1.e+6

        ts = np.arange(0, t_max-t_min)
        pop0 = populations[i] / normalization

        print(f'Place: {country} Population: {pop0/1e+6} M')

        title = ' '.join(countries) + ' ' + str(n_smooth) + ' day average'
        if shift_peak:
            title += ' peak centered'

        plt.title(title)

        # Loop over several columns (deaths, velocity, cases or else)
        for select in columns:
            values = data[data[select] == country].values

            if do_bin:
                bin_df = f.bin_mean(values)
                values = bin_df['mean']
                ts = bin_df['t']
        
            # We usually want to normalize by the population value but also find the peak
            if normalize:
                values = data[select].values[0:t_max-t_min]
                
                # Time interval
                d_t = t_max - t_min

                # Set the correct date values when dealing with regions or countries
                if do_type == 'countries':
                    times = data['date'].values[::-1][0:d_t]
                elif do_type == 'regions':
                    times = data['date'].values[:d_t]
                elif do_type == 'states':
                    times = data['date'].values[::-1][t_min:][::-1]

                values0 = np.max(values[np.logical_not(np.isnan(values))])
                print(f'N Values: {len(values)}, max:{values0}')

                # Find peak value (in case we want to rescale the curves at the peak)
                values = values/values0
                t_value = np.where(values == 1.0)
                values = values*values0
                values = values/pop0

                # Do some statistics
                total = np.sum(values[np.isfinite(values)])
                median = np.median(values[np.isfinite(values)])
                stddev = np.std(values[~np.isnan(values)])
                skewne = scipy.stats.skew(values[np.isfinite(values)])
                kurtos = scipy.stats.kurtosis(values[np.isfinite(values)])
                excess = kurtos - 3.0 
            
                # Keep track of some values
                medians.append(median)
                totals.append(total)

                print(f'Normalized , median: {median} std: {stddev} skew: {skewne} kurtosis: {kurtos} excess: {excess} for {country}')

            # In case we want to print stuff, get the right label
            data_label = 'data_'+select+'_'+country
            fit_label = 'fit_'+select+'_'+country
            mc_label = 'mc_'+select+'_'+country

            if shift_peak:
                t_shift = t_value[0][0]
                ts[:] -= t_shift

            n_labels = int((t_max - t_min) / 7) -1
            
            t_labels = []
            i_labels = []

            print(len(times), n_labels, n_labels * 7, t_max, t_min)

            for i in range(0, n_labels):
                if do_type == 'countries':
                    this_t = datetime.datetime.strptime(times[i * 7], '%m/%d/%y')
                elif do_type == 'regions':
                    this_t = str(times[i * 7]).replace('T', '-')
                    this_t = datetime.datetime.strptime(this_t, '%Y-%m-%d-%H:%M:%S')
                elif do_type == 'states':
                    this_t = str(times[i * 7])
                    this_t = datetime.datetime.strptime(this_t, '%Y%m%d')

                this_t = datetime.datetime.strftime(this_t, '%d') + '/' + datetime.datetime.strftime(this_t, '%m')
                t_labels.append(this_t) 
                i_labels.append(i * 7)

            if invert:
                plt.xticks(i_labels, t_labels[::-1], rotation='vertical')
                plt.plot(ts-n_smooth, values[::-1], label=data_label)
            else:
                plt.xticks(i_labels, t_labels, rotation='vertical')
                plt.plot(ts, values, label=data_label)

            # In case we want to add some analytical function to this mess
            if do_gompertz:
                g_mc, g_fit = f.fit_gompertz(x=ts, y=values, montecarlo=True)
                plt.plot(ts, g_fit, label=fit_label)
    
                # Montecarlo parameter estimation
                if montecarlo:
                    plt.plot(ts, g_mc, label=mc_label) 

    plt.legend()
    plt.xlabel('Day')
    plt.ylabel(columns[0] + ' per million')
    plt.tight_layout()

    if show:
        plt.show()

    else:
        plt.cla()
        plt.clf()
        plt.close()

    return medians, totals


if __name__ == "__main__":
    """ The main is a wrapper to select the kind of analysis and compare curves of regions or countries """

    do_type = 'countries'
    #do_type = 'regions'
    #do_type = 'states'
 
    # Select columns for the analysis
    columns = ['deaths_smooth']
    #columns = ['confirmed_smooth']
    #columns = ['deaths_acceleration']
    #columns = ['confirmed_velocity']
    
    if do_type == 'countries':
        #countries = ['Sweden', 'Italy']
        #countries = ['Sweden', 'Finland', 'Norway', 'Japan', 'Austria', 'Switzerland', 'Germany', 'Spain', 'New Zealand']
        #countries = ['Finland', 'Norway', 'Japan', 'New Zealand', 'Australia', 'Israel', 'Sweden', 'Germany', 'Italy']
        #countries = ['France', 'Germany', 'Italy', 'Belgium', 'Sweden', 'United Kingdom', 'Brazil']
        #countries = ['Italy', 'Belgium', 'Sweden', 'Uruguay', 'Brazil', 'Peru', 'Norway', 'Finland', 'Israel', 'Argentina', 'Germany', 'Poland', 'Greece', 'Spain', 'Portugal', 
        #'Japan', 'Vietnam', 'Luxembourg', 'United Kingdom', 'Slovenia', 'Serbia', 'Ukraine', 'Colombia', 'Turkey', 'Russia', 'Denmark', 'Malta' , 'Switzerland', 'Austria'] 

        #countries = ['Brazil', 'Italy', 'Belgium', 'France', 'Sweden', 'Chile', 'Israel']
        countries = ['Brazil', 'Colombia', 'Argentina', 'Chile', 'Peru', 'Portugal',  'Spain']
        #countries = ['Japan', 'Vietnam', 'Laos', 'Cambodia']
        #countries = ['Thailand', 'Vietnam', 'Laos', 'Cambodia']

    # US States
    elif do_type == 'states':
        countries = ['Florida', 'California'] #, 'North Dakota', 'South Dakota']
        #countries = ['Florida', 'California', 'North Dakota', 'South Dakota']

    # Italian regions
    elif do_type == 'regions':
        # TODO: fix the regions! Put some dictionary to correct the names when dealing with MOBILITY DATA
        #do_regions = True
        countries = ['Abruzzo', 'Lazio', 'Liguria', 'Veneto', 'Sicilia']
        #countries.append('Lazio')
        #countries.append('Piemonte')
        #countries.append('Veneto')
        #countries.append('Lombardia')
        #countries.append('Sardegna')
   
    # Initialize data, scraping stuff from the web if needed
    rd.init_data()

    # Set some parameters
    n_smooth = 14
    t_min = 50
    t_max =  400

    invert = True
    show = True

    # Run the program
    median_daily, tot_per_million = compare_curves(countries=countries, columns=columns, n_smooth=n_smooth, t_max=t_max, t_min=t_min, show=show, do_type=do_type, invert=invert)
    #median_daily, tot_per_million = compare_curves(countries=countries, columns=columns, n_smooth=n_smooth, t_max=t_max, t_min=t_min, show=False)
    mobility_daily, median_mobility = rd.mobility(countries=countries, do_type=do_type, day_init=t_min, day_end=t_max)

    masks, responses, stringencies = rd.country_data(countries=countries, verbose=True, day_start=t_min)

    #values_x = median_mobility; plt.xlabel('Median daily mobility baseline reduction %')
    values_x = masks; label_x = 'MaskWearing' 
    #values_x = responses; label_x = 'GovernmentResponse'
    values_y = tot_per_million; label_y = f'Median daily {columns[0]} per million'

    #indexes = np.where(np.array(values_x) > 0)[0]
    #values_x = np.array(values_x)[indexes]
    #values_y = np.array(values_y)[indexes]
    #countries = np.array(countries)[indexes]

    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.scatter(values_x, values_y)
    pears = scipy.stats.pearsonr(values_x, values_y)
    corr = np.corrcoef(values_x, values_y)

    print(f'Pearson correlation {label_x} vs. {columns[0]}: {pears}, correlation: {corr}')

    full_data = pd.DataFrame()
    full_data[columns[0]] = median_daily
    full_data['Masks'] = masks
    full_data['Responses'] = responses
    full_data['Stringency'] = stringencies
    full_data['Mobility'] = median_mobility

    print(full_data.corr())   


    for i, txt in enumerate(countries):
    
        plt.text(values_x[i], values_y[i], txt)
        print(f'{txt} Mask={masks[i]} Response={responses[i]} Stringency={stringencies[i]}')

    '''
    plt.tight_layout()
    plt.show()
    '''

    # Done
    exit()



