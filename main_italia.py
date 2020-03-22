import numpy as np
import read_io as rc
import analysis as al
import matplotlib.pyplot as plt

'''
    Base data settings
'''

base_data = '/home/edoardo/CoronaVirus/data/Italia/COVID-19/'
f_type = 'italia'

month_start = 2
month_end = 3

col = 10;   title = 'Casi Totali: Italia'; fout = 'casi_totali_'
#col = 6;   title = 'Casi Nuovi: Italia'; fout = 'casi_nuovi_'
#col = 9;   title = 'Casi Nuovi: Italia'; fout = 'decessi_'

output_name = fout +  'Italia.png'

#R = 3.57; t0 = 14.0
R = 2.95; t0 = 15.0


'''
    Init program
'''

data = rc.read_all_files(base_data, month_start, month_end, f_type)

n_pts = len(data)
print('Trovati dati per ', n_pts, ' giorni.')


'''
    Do some analysis (fit & parameter estimation)
'''

[giorni, dati_ordinati] = rc.get_data_italia(data, col)

n0 = dati_ordinati[0]

[dati_exp, cum_exp, pred] = al.simple_exp(R, t0, n_pts, n0)

sig = al.error_prediction(dati_ordinati, dati_exp)
#print(al.error_prediction(dati_ordinati, dati_exp))

print('Prediction + error: ', sig * pred + pred)

'''
    Plot data   
'''
plt.plot(giorni, dati_ordinati)

if col == 7:
    plt.plot(giorni, dati_exp)

elif col == 10:
    plt.plot(giorni, cum_exp)

elif col == 9:
    plt.plot(giorni, cum_exp)

plt.title(title)
plt.xlabel('Giorno')
plt.ylabel('N Casi')
plt.savefig(output_name)



