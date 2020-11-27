import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

import region_names

# Paese
#shape = '/home/edoardo/Udemy/DataScience/Maps/Italia/gadm36_ITA_0.shp'
# Regioni
shape = '/home/edoardo/Udemy/DataScience/Maps/Italia/gadm36_ITA_1.shp'
# Province
#shape = '/home/edoardo/Udemy/DataScience/Maps/Italia/gadm36_ITA_2.shp'
# Comuni
#shape = '/home/edoardo/Udemy/DataScience/Maps/Italia/gadm36_ITA_3.shp'

map_it = gpd.read_file(shape)
#print(map_it.head())
#print(map_it['NAME_1'])

regioni = '/home/edoardo/CoronaVirus/data/Italia/COVID-19/dati-regioni/dpc-covid19-ita-regioni-latest.csv'

dati = pd.read_csv(regioni)
#print(dati.head())
#print(dati['deceduti'])
#print(dati['denominazione_regione'])

#map_it.plot()

plt.show()
