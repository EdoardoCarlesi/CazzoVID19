#import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as so

'''
    DEFINIZIONE DI FUNZIONI GENERALI
'''

def new_cases(x):
    n_x = len(x)
    x_new = np.zeros((n_x - 1))

    for i in range (1, n_x):
        print(i, x[i], x[i-1])
        x_new[i-1] = x[i] - x[i-1]
        
    return x_new


def simple_exp(t, n0, r0):
    t0 = 15
    f0 = n0 * np.exp((r0 - 1)/t0 * t)
        
    return f0


'''
    INIZIO PROGRAMMA PRINCIPALE   
'''

regioni_csv = '/home/edoardo/CoronaVirus/data/World/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(regioni_csv)

hd = df.head(0)
head = hd.columns
n_days = len(head) - 4
days = np.arange(1, n_days)

select_col = 'Country/Region'

all_countries = df[select_col].unique()

offset = 25

countries = ['Japan', 'Serbia', 'Spain', 'Italy']

plt.title(select_col)
plt.xscale('log')
plt.yscale('log')
plt.ylim([1, 2.e+4])

for country in countries:
    confirmed = df[df[select_col] == country]
    this_row = confirmed.iloc[0]
    #print(this_row, type(this_row))
    #y_data = this_row[5:n_days-1]
    #print(confirmed)
    #print(y_data[0:10])

    y_data = pd.to_numeric(this_row[4:n_days+3])
    diff_data = new_cases(y_data)
    #plt.plot(days, y_data, label = country, marker = 'o')
    #print(diff_data, len(y_data))
    #plt.plot(days[1+offset:n_days], diff_data[offset:n_days], label = country, marker = 'o')
    plt.scatter(days[1+offset:n_days], diff_data[offset:n_days], label = country, marker = 'o')

plt.legend()
plt.show()


'''

deceduti = df['deceduti'][df['denominazione_regione'] == 'Abruzzo']
totale_casi= df['totale_casi'][df['denominazione_regione'] == 'Abruzzo']

n_days = len(totale_casi)
days = np.arange(1, n_days+1)

#column_name = 'deceduti'
column_name = 'nuovi_positivi'

#plt.xscale('log')
plt.yscale('log')
plt.title(column_name)

for regione in nomi_regioni:
    values = df[column_name][df['denominazione_regione'] == regione]
    plt.plot(days, values, label=regione)

#plt.legend([0.1, 0.7, 0.3, 0.9])
#plt.legend('upper left')
plt.legend()
plt.show()

'''
