# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:47:55 2020

@author: dp
"""
# Riparian User Matrix (location)
import pandas as pd
import numpy

flow_table_df = pd.read_csv('input/flows.csv')
# flow_table_df = pd.read_csv('basins_users_MS_HW/basins_flows_to_MS_HW.csv')

basins = flow_table_df["BASIN"].tolist()

rip_user_df = pd.read_csv('input/riparian_demand.csv')
# rip_user_df = pd.read_csv('basins_users_MS_HW/riparian_users_MS_HW.csv')

rip_user = rip_user_df["USER"].tolist()
user_location = rip_user_df["BASIN"].tolist()
    
basin_use = {rip_user[i] : user_location[i] for i, user in enumerate(rip_user)}  
index_dictionary = {basins[k] : [k] for k, basin in enumerate(basins)}

user_matrix = numpy.zeros([numpy.size(rip_user), numpy.size(basins)], dtype = int)

for i, user in enumerate(rip_user):
    user_matrix[i][index_dictionary[basin_use[user]]] = 1

user_matrix = user_matrix.transpose()   

riparian_user_matrix = pd.DataFrame(user_matrix, index = basins)
riparian_user_matrix.index.name = "BASIN"

rip_user = rip_user_df["USER"].tolist()
riparian_user_matrix.columns = rip_user

riparian_user_matrix.to_csv("input/riparian_user_matrix.csv")
# riparian_user_matrix.to_csv("input_MS_HW/riparian_user_matrix_MS_HW.csv")

##########################################################################################################################

# User connectivity matrix (1 if user is upstream of basin)
basin_connectivity_matrix = pd.read_csv("input/basin_connectivity_matrix.csv", index_col = "BASIN")
# basin_connectivity_matrix = pd.read_csv("input_MS_HW/basin_connectivity_matrix_MS_HW.csv", index_col = "BASIN")

basin_connectivity_matrix = numpy.array(basin_connectivity_matrix)

user_connectivity = numpy.matmul(basin_connectivity_matrix.transpose(), user_matrix)

user_connectivity = pd.DataFrame(user_connectivity, index =basins)

rip_user = rip_user_df["USER"].tolist()
user_connectivity.columns= rip_user
user_connectivity.index.name = "BASIN"

user_connectivity.to_csv("input/riparian_user_connectivity_matrix.csv")
# user_connectivity.to_csv("input_MS_HW/riparian_user_connectivity_matrix_MS_HW.csv")

