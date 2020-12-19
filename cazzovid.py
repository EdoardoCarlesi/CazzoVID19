import pandas as pd
import numpy as np
import functions as f
import read_data as rd

import seaborn as sns
import matplotlib.pyplot as plt
import scipy

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


def compare_curves(countries=None, normalize=True, columns=None, n_smooth=7, t_max=320, t_min=250, n_days=1, show=True):
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
    
    # TODO avoid this loop on countries and replace with 
    for i, country in enumerate(countries):

        if do_countries:
            data = rd.extract_country(n_days=n_days, smooth=n_smooth, country=country)     
            #data = data.dropna()
        else:
            data = rd.extract_region(region=country, smooth=n_smooth)
            #data = data.dropna()

        t_max = t_max #- n_smooth
        ts = np.arange(0, t_max-t_min)
        pop0 = populations[i]

        print(f'Place: {country} Population: {pop0/1e+6} M')

        title = ' '.join(countries) + ' ' + str(n_smooth) + ' day average'
        if shift_peak:
            title += ' peak centered'

        plt.title(title)

        for select in columns:
            values = data[data[select] == country].values

            if do_bin:
                bin_df = f.bin_mean(values)
                values = bin_df['mean']
                ts = bin_df['t']

            if normalize:
                #values = data[select].values[0:t_max-t_min][::-1]
                values = data[select].values[0:t_max-t_min]
                values0 = np.max(values[np.logical_not(np.isnan(values))])
                print(f'N Values: {len(values)}, max:{values0}')

                values = values/values0
                t_value = np.where(values == 1.0)
                values = values*values0
                values = values/pop0

                total = np.sum(values[np.isfinite(values)])
                median = np.median(values[np.isfinite(values)])
                stddev = np.std(values[~np.isnan(values)])
                skewne = scipy.stats.skew(values[np.isfinite(values)])
                kurtos = scipy.stats.kurtosis(values[np.isfinite(values)])
                excess = kurtos - 3.0 
                print(f'Normalized , median: {median} std: {stddev} skew: {skewne} kurtosis: {kurtos} excess: {excess} for {country}')
                #sns.kdeplot(values[np.isfinite(values)])
                #plt.show()
            
                medians.append(median)
                totals.append(total)

            data_label = 'data_'+select+'_'+country
            fit_label = 'fit_'+select+'_'+country
            mc_label = 'mc_'+select+'_'+country

            if shift_peak:
                t_shift = t_value[0][0]
                
                ts[:] -= t_shift

            plt.plot(ts, values, label=data_label)
            #sns.kdeplot(values[np.isfinite(values)])
            #sns.distplot(values[np.isfinite(values)])

            if do_gompertz:
                g_mc, g_fit = f.fit_gompertz(x=ts, y=values, montecarlo=True)
                plt.plot(ts, g_fit, label=fit_label)

                if montecarlo:
                    plt.plot(ts, g_mc, label=mc_label) 

    plt.legend()
 
    if show:
        plt.show()

    else:
        plt.cla()
        plt.clf()
        plt.close()

    return medians, totals

    '''
        plt.show(block=False)
        plt.pause(7)
        plt.cla()
        plt.clf()
        plt.close()
    '''


if __name__ == "__main__":
    """ The main is a wrapper to select the kind of analysis and compare curves of regions or countries """

    '''
    countries = ['Sweden', 'Italy', 'Germany', 'Austria']; do_regions=False
    #countries.append('Peru')
    countries.append('Switzerland')
    countries.append('Norway')
    countries.append('Finland')
    countries.append('Belgium')
    countries.append('Russia')
    countries.append('Greece')
    countries.append('Serbia')
    countries.append('Slovenia')
    countries.append('Spain')
    #countries.append('France')
    countries.append('Poland')
    countries.append('Japan')
    countries.append('Portugal')
    countries.append('Hungary')
    countries.append('Brazil')
    #countries.append('Peru')
    #countries.append('Chile')
    #countries.append('Argentina')
    #countries = ['Switzerland', 'Hungary', 'Austria']
    '''

    # TODO: fix the regions! Put some dictionary to correct the names
    countries = ['Abruzzo']; do_regions = True

    #columns = ['confirmed_variation_smooth', 'confirmed_smooth']
    #columns = ['confirmed_smooth', 'confirmed']
    #columns = ['confirmed_smooth', 'deaths_smooth']
    #columns = ['deaths']
    #columns = ['confirmed_smooth']
    columns = ['deaths']
    #columns = ['deaths_variation', 'deaths_smooth']
    #columns = ['confirmed_variation', 'confirmed_smooth']
    #columns = ['confirmed_acceleration']
    #columns = ['deaths_acceleration']
    #columns = ['confirmed_velocity']
    
    # Initialize data, scraping stuff from the web if needed
    rd.init_data()

    # Set some parameters
    n_smooth = 7
    t_min = 40
    t_max =  325

    # Run the program
    meds, tots = compare_curves(countries=countries, columns=columns, n_smooth=n_smooth, t_max=t_max, t_min=t_min, show=False)
    mobilities, medians = rd.mobility(countries=countries, do_regions=do_regions)

    values = tots

    plt.scatter(medians, values)

    for i, txt in enumerate(countries):
        plt.text(medians[i], values[i], txt)

    plt.tight_layout()
    plt.show()

    # Done
    exit()



