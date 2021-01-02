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


def compare_curves(countries=None, normalize=True, columns=None, n_smooth=7, t_max=320, t_min=250, n_days=1, show=True, invert=True):
    """ Fit data from various countries/regions to Gompertz curve """

    # Median and total values for the indicator which will be returned
    medians = []
    totals = []

    # What kind of analysis?
    montecarlo = False
    do_gompertz = False
    shift_peak = False
    do_bin = False
    
    # Well this implementatation is not elegant at all...
    populations = rd.people_per_region(regions=countries)

    # If we find and empty list then it means we are not dealing with regions
    if populations != []:
        print('Analyzing regions...')
        do_countries = False

    # Try analyzing countries instead
    else:
        populations = rd.people_per_country(countries=countries) 

        # Again, if we find a non empty list it means we can proceed
        if populations != []:
            print('Analyzing countries...')
            do_countries = True
        else:
            print(f'The countries/regions mentioned are not available. Exit program...')
            exit()
    
    # Loop on countries we want to check
    for i, country in enumerate(countries):

        if do_countries:
            data = rd.extract_country(n_days=n_days, smooth=n_smooth, country=country)     
        else:
            data = rd.extract_region(region=country, smooth=n_smooth)

        ts = np.arange(0, t_max-t_min)
        pop0 = populations[i] / 1.e+6

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

                # Set the correct date values when dealing with regions or countries
                if do_countries:
                    times = data['date'].values[::-1][0:t_max-t_min]
                else:
                    times = data['date'].values[::-1][t_max-t_min::]

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

            n_labels = int((t_max - t_min) / 7)
            
            t_labels = []
            i_labels = []

            for i in range(0, n_labels):
                if do_countries:
                    this_t = datetime.datetime.strptime(times[i * 7], '%m/%d/%y')
                else:
                    this_t = str(times[i * 7]).replace('T', '-')
                    this_t = datetime.datetime.strptime(this_t, '%Y-%m-%d-%H:%M:%S')

                this_t = datetime.datetime.strftime(this_t, '%d') + '/' + datetime.datetime.strftime(this_t, '%m')
                t_labels.append(this_t) 
                i_labels.append(i * 7)

            if invert:
                plt.xticks(i_labels, t_labels[::-1], rotation='vertical')
                plt.plot(ts[::-1], values, label=data_label)
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
    plt.ylabel('New cases per million')
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

    #countries = ['Abruzzo', 'Lombardia', 'Lazio', 'Veneto', 'Campania']
    #countries = ['Sweden', 'Italy']
    #countries = ['Italy', 'Belgium', 'Sweden', 'Uruguay', 'Brazil', 'Peru', 'Norway', 'Finland', 'Israel', 'Slovakia', 'Argentina', 'Chile', 'Germany', 'Poland', 'Greece', 'Spain', 'Portugal', 
    #        'Japan', 'Vietnam', 'Luxembourg', 'United Kingdom', 'Slovenia', 'Serbia', 'Ukraine', 'Belarus', 'Colombia', 'Turkey', 'Russia', 'Denmark', 'Malta' , 'Switzerland', 'Austria', 
    #        'Croatia', 'Mexico', 'Armenia', 'Qatar', 'Panama', 'France', 'Moldova']
    #countries = ['Italy', 'Czechia', 'Slovakia', 'Germany', 'Belgium', 'Sweden'] 
    #countries = ['Italy', 'Belgium', 'Norway', 'Finland', 'Slovakia', 'Germany', 'France', 'Netherlands'] 
    countries = ['Sweden', 'Italy', 'Germany', 'Ireland']
    #countries = ['Switzerland', 'Hungary', 'Austria']

    do_regions = False

    #countries.append('Japan')
    #countries.append('Sweden')
    #countries.append('Denmark')
    #countries.append('Serbia')
    #countries.append('France')
    #countries.append('Slovakia')
    #countries.append('Slovenia')
    #countries.append('Italy')

    #countries.append('Peru')
    #countries.append('Switzerland')
    #countries.append('Norway')
    #countries.append('Finland')
    #countries.append('Belgium')
    #countries.append('Denmark')

    # TODO: fix the regions! Put some dictionary to correct the names when dealing with MOBILITY DATA
    #do_regions = True
    #countries = ['Abruzzo', 'Lazio', 'Liguria']
    #countries = ['Abruzzo']; do_regions = True
    #countries.append('Lazio')
    #countries.append('Piemonte')
    #countries.append('Veneto')
    #countries.append('Lombardia')
    #countries.append('Sardegna')

    #columns = ['confirmed_variation_smooth', 'confirmed_smooth']
    #columns = ['confirmed_smooth', 'confirmed']
    #columns = ['confirmed_smooth', 'deaths_smooth']
    #columns = ['deaths_smooth']
    columns = ['confirmed_smooth']
    #columns = ['confirmed_velocity']
    #columns = ['confirmed_acceleration']
    #columns = ['deaths_smooth']
    #columns = ['deaths_variation', 'deaths_smooth']
    #columns = ['confirmed_variation', 'confirmed_smooth']
    #columns = ['confirmed_acceleration']
    #columns = ['deaths_acceleration']
    #columns = ['confirmed_velocity']
    
    # Initialize data, scraping stuff from the web if needed
    rd.init_data()

    # Set some parameters
    n_smooth = 7
    t_min = 250
    t_max =  340

    # Run the program
    meds, tots = compare_curves(countries=countries, columns=columns, n_smooth=n_smooth, t_max=t_max, t_min=t_min, show=True)
    median_daily, tot_per_million = compare_curves(countries=countries, columns=columns, n_smooth=n_smooth, t_max=t_max, t_min=t_min, show=False)
    mobility_daily, median_mobility = rd.mobility(countries=countries, do_regions=do_regions, day_init=t_min, day_end=t_max)

    masks, responses, stringencies = rd.country_data(countries=countries, verbose=True, day_start=t_min)

    #values_x = median_mobility; plt.xlabel('Median daily mobility baseline reduction %')
    #values_x = masks; plt.xlabel('Mask rules index')
    values_x = responses; plt.xlabel('Government Response')
    values_y = tot_per_million; plt.ylabel(f'Median daily {columns[0]} per million')

    plt.scatter(values_x, values_y)

    for i, txt in enumerate(countries):
        print(f'{txt} Mask={masks[i]} Response={responses[i]} Stringency={stringencies[i]}')
        plt.text(values_x[i], values_y[i], txt)

    plt.tight_layout()
    #plt.show()

    # Done
    exit()



