#import seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


regioni_csv = '/home/edoardo/CoronaVirus/data/Italia/COVID-19/dati-regioni/dpc-covid19-ita-regioni.csv'
df = pd.read_csv(regioni_csv)

hd = df.head(0)
head = hd.columns

for h in head:
    print(h)

#nomi_regioni = df['denominazione_regione'].unique()
#print(nomi_regioni)

nomi_regioni = ['Veneto', 'Lombardia', 'Emilia-Romagna', 'Lazio']

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
plt.plot()
plt.show()

