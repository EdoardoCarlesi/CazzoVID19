import numpy as np
import read_io as rc
import analysis as al
import matplotlib.pyplot as plt

'''
    Base data settings
'''

base_data = '/home/edoardo/CoronaVirus/data/Italia/COVID-19/'
#f_type = 'regioni'
f_type = 'italia'

month_start = 2
month_end = 3

#col = 11 # nuovi positivi

region = 2; region_name = 'Lombardia'
#region = 4; region_name = 'Veneto'
#region = 11; region_name = 'Lazio'
#region = 6; region_name = 'Liguria'
#region = 5; region_name = 'FriuliVeneziaGiulia'

#col = 11;   title = 'Nuovi Positivi: ' + region_name; fout = 'casi_nuovi_'
col = 14;   title = 'Casi Totali: ' + region_name; fout = 'casi_totali_'

output_name = fout + region_name + '.png'

# Parametri Lombardia
if region == 2:
    R = 2.4; t0 = 14.0

# Parametri Veneto
elif region == 4:
    R = 2.4; t0 = 14.0

# Parametri Lazio
elif region == 11:
    R = 1.9; t0 = 14.0

# Parametri Liguria
elif region == 6:
    R = 2.5; t0 = 14.0

else :
    R = 2.2; t0 = 14.0
    

'''
    Init program
'''

data = rc.read_all_files(base_data, month_start, month_end, f_type)

n_pts = len(data)
print('Trovati dati per ', n_pts, ' giorni.')


'''
    Do some analysis (fit & parameter estimation)
'''


[giorni, dati_ordinati] = rc.get_data_regioni(data, col)

#print(dati_ordinati[region, 0])
n0 = dati_ordinati[region, 0]

[dati_exp, cum_exp] = al.simple_exp(R, t0, n_pts, n0)

'''
    Plot data   
'''

#plt.yscale('log')
plt.plot(giorni, dati_ordinati[region])

if col == 11:
    plt.plot(giorni, dati_exp)

elif col == 14:
    plt.plot(giorni, cum_exp)

plt.title(title)
plt.xlabel('Giorno')
plt.ylabel('N Casi')
plt.savefig(output_name)



