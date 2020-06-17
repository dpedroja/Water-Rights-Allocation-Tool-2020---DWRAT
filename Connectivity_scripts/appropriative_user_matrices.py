# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 10:46:14 2020

@author: dp
"""

# Appropriative User Matrix (location)

import pandas as pd
import numpy

flow_table_df = pd.read_csv('input/flows.csv')
# flow_table_df = pd.read_csv('basins_users_MS_HW/basins_flows_to_MS_HW.csv')

basins = flow_table_df["BASIN"].tolist()

app_user_df = pd.read_csv('input/appropriative_demand.csv')
# app_user_df = pd.read_csv('basins_users_MS_HW/appropriative_users_MS_HW.csv')

app_user = app_user_df["USER"].tolist()
user_location = app_user_df["BASIN"].tolist()

basin_use = {app_user[i] : user_location[i] for i, user in enumerate(app_user)}  
index_dictionary = {basins[k] : [k] for k, basin in enumerate(basins)}
    
user_matrix = numpy.zeros([numpy.size(app_user), numpy.size(basins)], dtype = int)

for i, user in enumerate(app_user):
    user_matrix[i][index_dictionary[basin_use[user]]] = 1

user_matrix = user_matrix.transpose()     

# appropriative_user_matrix = pd.DataFrame(user_matrix)
# appropriative_user_matrix.insert(0, "BASIN", basins)

# app_user.insert(0, "BASIN")
# appropriative_user_matrix.columns = [app_user]




appropriative_user_matrix = pd.DataFrame(user_matrix, index = basins)
appropriative_user_matrix.index.name = "BASIN"

app_user = app_user_df["USER"].tolist()
appropriative_user_matrix.columns = app_user

appropriative_user_matrix.to_csv("input/appropriative_user_matrix.csv")
# appropriative_user_matrix.to_csv("input_MS_HW/appropriative_user_matrix_MS_HW.csv")

##########################################################################################################################

# User connectivity matrix (1 if user is upstream of basin)

basin_connectivity_matrix = pd.read_csv("input/basin_connectivity_matrix.csv", index_col = "BASIN")
# basin_connectivity_matrix = pd.read_csv("input_MS_HW/basin_connectivity_matrix_MS_HW.csv", index_col = "BASIN")

basin_connectivity_matrix = numpy.array(basin_connectivity_matrix)

user_connectivity = numpy.matmul(basin_connectivity_matrix.transpose(), user_matrix)

user_connectivity = pd.DataFrame(user_connectivity, index = basins)

app_user = app_user_df["USER"].tolist()
user_connectivity.columns= app_user

user_connectivity.index.name = "BASIN"

user_connectivity.to_csv("input/appropriative_user_connectivity_matrix.csv")
# user_connectivity.to_csv("input_MS_HW/appropriative_user_connectivity_matrix_MS_HW.csv")


