coord_data = 'world_country_and_usa_states_latitude_and_longitude_values.csv'
dataset = 'github-cs229-project/compressed_dataset'

### get country coordinates ###
import numpy as np
my_data_str = np.genfromtxt(coord_data, delimiter=',', skip_header=1, dtype=str)
print(my_data_str.shape)
my_data_str = my_data_str[:,1:4]
lats = [float(x) for x in my_data_str[:,0]]
longs = [float(x) for x in my_data_str[:,1]]
countries_with_coords = my_data_str[:,2]
coords = dict()
for i in range(len(countries_with_coords)):
    coords[countries_with_coords[i]] = (lats[i],longs[i])

### get list of countries from dataset ###
import os, os.path
data_countries = os.listdir(dataset)  # not used

# sort countries by top 20
# 675 pics in 20th-most-pics country (Norway)
data_countries = []
for country in os.listdir(dataset):
    if len(os.listdir(dataset+'/'+country)) < 675: continue
    data_countries.append(country)

# ensure we have the top 20 countries for the matrix
top_n_countries = 20
assert len(data_countries) == top_n_countries

### calculate distance matrix ###
n_countries = len(data_countries)
from geopy import distance
dist_matrix = np.zeros((n_countries,n_countries))
keys = coords.keys
for a,cunt_a in enumerate(data_countries):
    for b,cunt_b in enumerate(data_countries):
        if coords.get(cunt_a) is None or coords.get(cunt_b) is None: continue

        dist_matrix[a,b] = distance.great_circle(coords[cunt_a], coords[cunt_b]).miles

### Don't have coords (or therefore distances) for following countries ###
# Macao, Reunion, Palestine, Czechia, North Macedonia, Curacao

dist_matrix = np.array(dist_matrix)
# print(dist_matrix)

normalized_dist_matrix = dist_matrix / np.max(dist_matrix)
# print(normalized_dist_matrix)