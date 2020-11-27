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
        #print(i, x[i], x[i-1])
        x_new[i-1] = x[i] - x[i-1]
        
    return x_new


def relative_increase(x):
    n_x = len(x)
    x_new = np.zeros((n_x - 1))

    for i in range (1, n_x):
        #print(i, x[i], x[i-1])
        x_new[i-1] = (x[i] - x[i-1]) / x[i]
        
    return x_new


def simple_exp(all_t, offset, n0, r0):
    t0 = 15.0
    deltaT = len(all_t) - offset -1
    f0 = np.zeros((deltaT))

    for i in range(0, deltaT):
        f0[i] = n0 * np.exp((r0 - 1)/t0 * i)
        
    return f0


def interp(x):
    n = len(x)

    for i in range(1, n-1):
        if i > 2:
            x0 = x[i-1]
            x1 = x[i+1]

            if x[i] == 0:
                x[i] = 0.5 * (x0 + x1)
    return x


def lin_fit(x, a, b):
    
    return x * a + b


# We determine the exponential fit value over the first two weeks
def single_fit(x_all, y_all):
    a0 = 1.0
    t0 = 15.0
    tmax = 30

    n0 = np.log(y_all[0])
    logy = np.log(y_all[0:tmax])

    popt, pcov = so.curve_fit(lin_fit, x_all[:tmax], logy, p0=[n0, a0], method = 'lm')

    r0 = popt[0] * t0 + 1
    print('Approx R0: ', r0)

    return r0
    

def split_sample_fit(x_all, y_all, n_pre):
    n_split = n_pre[0]

    a0 = 1.0
    a1 = 1.0
    t0 = 15.0

    n0 = np.log(y_all[0])
    n1 = np.log(y_all[n_split])

    #print('YALL: ', y_all, type(y_all))
    logy0 = np.log(y_all[:n_split])
    logy1 = np.log(y_all[n_split:])

    popt0, pcov0 = so.curve_fit(lin_fit, x_all[:n_split], logy0, p0=[n0, a0], method='lm')   
    popt1, pcov1 = so.curve_fit(lin_fit, x_all[n_split:], logy1, p0=[n1, a1], method='lm')   

    r0 = popt0[0] * t0 + 1
    r1 = popt1[0] * t0 + 1

    '''
    print(popt0, r0)
    print(popt1, r1)
    '''

    # Convert to R0 parameter
    return [r0, r1]

#def smooth_average():
# TODO


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

offset = 30
n_min = 20

#countries = ['Japan', 'Serbia', 'Spain', 'Italy']
#countries = ['Serbia', 'Spain', 'Italy']
#countries = ['Italy']
countries = ['Spain']

x_lims = [offset, n_days]
y_lims = [n_min, 1.e+5]

plt.title(select_col)
#plt.xscale('log')
plt.yscale('log')
plt.xlim(x_lims)
plt.ylim(y_lims)

for country in countries:
    confirmed = df[df[select_col] == country]
    this_row = confirmed.iloc[0]

    # Filter out first descriptive columns
    y_data = pd.to_numeric(this_row[4:n_days+3])
    
    df = pd.DataFrame(data = y_data.values, columns = ['y'], copy = False)
    n0 = y_data[y_data > n_min]
    deltaN = len(y_data) - len(n0)

    # Total days taken into account
    offset = deltaN

    # Plot against a simple exponential model
    y_data_smooth = df['y'].rolling(window = 7).mean().values
    print('Smooth: ', y_data_smooth)
    r0 = single_fit(days[offset:], y_data_smooth[offset:])
    y_exp = simple_exp(days, offset, n0[0], r0)
    #print(y_exp)

    # Choose data type and interpolate for missing days / values
    diff_data = interp(new_cases(y_data))
    df2 = pd.DataFrame(data = diff_data, columns = ['use_data'], copy = True)
    data_smooth = df2['use_data'].rolling(window = 10).mean()

    #diff_data = relative_increase(y_data)

    # Determine deviations from the exponential regime
    exp_thr = 5.0
    #exp_dev = abs(data_smooth[offset:n_days] - y_exp) / diff_data[offset:]
    exp_dev = abs(diff_data[offset:n_days] - y_exp) / diff_data[offset:]
    exp_day = exp_dev[exp_dev > exp_thr]
    print(exp_dev)

    ind = np.where(exp_dev == exp_day[0])
    print('The deviation from the exponential regime happened on the: ', head[ind[0] + 4 + offset], ' in ', country, ', n= ', ind[0], ' days after.')
    
    # Determine days - interval range
    x_thr = ind[0] + offset
    use_days0 = days[1+offset:x_thr[0]]
    use_days1 = days[x_thr[0]:]
    use_days = days[1+offset:]
    use_data = diff_data[offset:]
    
    df3 = pd.DataFrame(data = use_data, columns = ['use_data'], copy = True)
    use_data_smooth = df3['use_data'].rolling(window = 5).mean()
    
    # Fit and check the best fit parameters
    [R0, R1] = split_sample_fit(use_days, use_data, ind[0])

    # Vertical line for the actual plots
    x_line = [x_thr, x_thr]; y_line = y_lims


    # Separate the two plots
    N0 = (0.66 * n0[0] + 0.33 * n0[1])
    y_exp0 = simple_exp(days[:x_thr[0]], offset, N0, R0)
    
    N1 = diff_data[x_thr[0]+1] 
    y_exp1 = simple_exp(days, x_thr[0], N1, R1)

    # Do the actual plots
    #plt.scatter(days[1+offset:n_days], diff_data[offset:n_days], label = country, marker = 'o')
    plt.plot(use_days, use_data_smooth, label = country, marker = 'o')
    plt.plot(use_days, use_data, label = country, marker = 'o')
    plt.plot(use_days, y_exp, label = 'R0 = ' + str(r0))
    plt.plot(use_days0, y_exp0, label = 'R0 = ' + str(R0))
    plt.plot(use_days1[1:], y_exp1, label = 'R0 = ' + str(R1))
    plt.plot(x_line, y_line, color='black')

    plt.plot()
    #print(exp_day)
    #print(diff_data[offset:])

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
