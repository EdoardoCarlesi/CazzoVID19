import numpy as np

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




