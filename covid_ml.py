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


def compare_curves(do_countries=False, do_regions=True, countries=[], normalize=True):
    """ Fit data from various countries/regions to Gompertz curve """

    # How many days forward for the target and on how many days should we smooth (rolling average)
    n_days = 1
    n_smooth = 13

    # What kind of analysis?
    montecarlo = False
    do_gompertz = False
    shift_peak = False
    do_bin = False

    if do_countries:
        #countries = ['Sweden', 'Italy']
        #countries = ['Italy', 'Belgium', 'Serbia']; populations = [62e+6, 11.5e+6, 7.5e+6]
        countries = ['Italy', 'Czechia', 'Slovakia', 'Germany', 'Belgium', 'Sweden'] 
        #countries = ['Italy', 'Belgium', 'Norway', 'Finland', 'Slovakia', 'Germany'] 
        #countries = ['Sweden', 'Italy', 'Belgium', 'Serbia']
 
    elif do_regions:
        countries = ['Sardegna', 'Friuli Venezia Giulia', 'Lazio', 'Abruzzo'] 
        populations = [rd.people_per_region(region=reg) for reg in countries]
    
        print(populations)
        #countries = ['Abruzzo', 'Lombardia', 'Lazio', 'Veneto', 'Campania']
        #countries = ['Abruzzo', 'Lombardia', 'Lazio', 'Sicilia', 'Veneto', 'Campania']

    #columns = ['confirmed_smooth', 'deaths_smooth']
    #columns = ['confirmed_variation_smooth', 'confirmed_smooth']
    #columns = ['confirmed_smooth', 'confirmed']
    #columns = ['confirmed_smooth', 'deaths_smooth']
    columns = ['deaths']
    #columns = ['deaths_variation', 'deaths_smooth']
    #columns = ['confirmed_variation', 'confirmed_smooth']
    #columns = ['confirmed_smooth']
    #columns = ['confirmed_acceleration']
    #columns = ['deaths_acceleration']
    #columns = ['confirmed_velocity']
    #columns = ['deaths_smooth']
    #columns = ['confirmed_variation_smooth']
    #columns = ['confirmed_variation_smooth']
    #columns = ['confirmed_variation_smooth', 'deaths_variation_smooth']

    for i, country in enumerate(countries):

        if do_countries:
            data = rd.extract_country(n_days=n_days, smooth=n_smooth, country=country)     
            data = data.dropna()
        elif do_regions:
            data = rd.extract_region(region=country, smooth=n_smooth)
            #data = data.dropna()

        t_min = 220
        t_max = 320 - n_smooth
        ts = np.arange(0, t_max-t_min)
        pop0 = populations[i]
        print(f'Place: {country} Population: {pop0/1e+6} M')

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

            if normalize:
                values /= values0
                t_value = np.where(values == 1.0)
                values *= values0
                values /= pop0
                #values *= 1.0e+6

                median = np.median(values[np.isfinite(values)])
                stddev = np.std(values[~np.isnan(values)])
                skewne = scipy.stats.skew(values[np.isfinite(values)])
                kurtos = scipy.stats.kurtosis(values[np.isfinite(values)])
                excess = kurtos - 3.0 
                print(f'Normalized , median: {median} std: {stddev} skew: {skewne} kurtosis: {kurtos} excess: {excess} for {country}')
                #sns.kdeplot(values[np.isfinite(values)])
                #plt.show()

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
    plt.show() 
    #block=False)
    #plt.pause(7)
    #plt.close()


if __name__ == "__main__":

    """ MAIN PROGRAM """

    #countries = country_data()

    #compare_curves(do_countries=True, do_regions=False)
    compare_curves()





