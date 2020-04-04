import numpy as np
import scipy.optimize as so



def simple_exp(R, t0, n_day, n0):
    n_tot = np.zeros((n_day))
    n_cum = np.zeros((n_day))

    if n0 == 0:
        n0 = 1

    n_cum[0] = n0
    for d in range(0, n_day):
        fac = (R - 1) * (d / t0)
        n_tot[d] = n0 * np.exp(fac)
        if d > 0:
            n_cum[d] = n_cum[d-1] + n_tot[d]
        #print(d, n_tot[d])

    fac = (R - 1) * ( (n_day+1) / t0)
    predicted = n0 * np.exp(fac)
    print('Prediction for day+1: ', predicted)
    fac = (R - 1) * ( (n_day+2) / t0)
    predicted = n0 * np.exp(fac)
    print('Prediction for day+2: ', predicted)
    fac = (R - 1) * ( (n_day+3) / t0)
    predicted = n0 * np.exp(fac)
    print('Prediction for day+3: ', predicted)
    fac = (R - 1) * ( (n_day+4) / t0)
    predicted = n0 * np.exp(fac)
    print('Prediction for day+4: ', predicted)

    return [n_tot, n_cum, predicted]


def exp(t, n0, mu0):

    #mu = (R - 1) / t0
    #mu0 = (1.5) / 14.0
    
    e = n0 * np.exp(t * mu0)

    return e


def exp0(t, mu0):

    #mu = (R - 1) / t0
    #mu0 = (1.5) / 14.0
    n0 = 229.0

    e = n0 * np.exp(t * mu0)

    return e

def linear(x, A, B):
    return A * x + B


def linearize_fit(xdata, ydata, params):

    n_pts = len(xdata)
    lnx = np.zeros((n_pts))
    lny = np.zeros((n_pts))

    for i in range(0, n_pts):
        lnx[i] = np.log(xdata[i])
        lny[i] = np.log(ydata[i])

    popt, pcov = so.curve_fit(linear, lnx, lny)  

    print(popt)

    n0 = np.exp(popt[0])
    mu = np.exp(popt[1])

    print('LinFit: ', n0, mu)


def fit_exp(xdata, ydata, params):

    popt, pcov = so.curve_fit(exp, xdata, ydata) #, p0=params, method='lm')
    #popt, pcov = so.curve_fit(exp0, xdata, ydata, p0=params[1], method='lm')
    print('Init params: ', params)
    #popt, pcov = so.curve_fit(exp, xdata, ydata) #, p0=params, method='lm')
    

    n0 = params[0]
    #n0 = 229.0
    t0 = 14.0
    R0 = t0 * (params[1]) + 1
    nf = popt[0]
    #Rf = t0 * (popt[1]) + 1
    Rf = t0 * (popt) + 1


    #print('Popt: ', popt) #' pcov: ', pcov)
    #print('n0: ', n0, ' nf: ', nf, popt[0], ' R0: ', R0, ' Rf: ', Rf, popt[1])
    #print('n0: ', n0, ' nf: ', nf, ' R0: ', R0, ' Rf: ', Rf, popt[0])
    
    return [t0, nf, Rf]

def moving_average(data, n_avg):

    w0 = 0.75
    w1 = 0.25
    w2 = 0.33

    n_pts = len(data)

    for i in range(0, n_pts):
        if i == 0:
            new_pt = (data[0] * w0 + data[1] * w1)
        elif i == n_pts -1:
            new_pt = (data[i] * w0 + data[i-1] * w1)
        else:
            new_pt = w2 * (data[i] + data[i-1] + data[i+1])


def error_prediction(data_real, data_simu):
    
    n_pts = len(data_simu)

    s = 0
    n_pts_real= n_pts
    for i in range(0, n_pts):

        if data_real[i] == 0:
            n_pts_real -= 1
        else:
            d = (data_simu[i] - data_real[i]) / data_real[i]
            s += d * d

    return np.sqrt(s) / n_pts_real



def mutiple_R_exp(Rs, t0s, n_days):
    # Rs & t0s, n_days sono degli array con piu' valori corrispondenti alle diverse fasi
    
    tot_days = 0

    for d in n_days:
        tot_days += d





