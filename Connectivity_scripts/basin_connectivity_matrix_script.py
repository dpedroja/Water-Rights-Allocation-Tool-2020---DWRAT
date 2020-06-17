# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:48:44 2020

@author: dpedroja
"""

# Basin Connectivity Matrix

import pandas as pd
import numpy
flow_table_df = pd.read_csv('input/flows.csv')
# flow_table_df = pd.read_csv('basins_users_MS_HW/basins_flows_to_MS_HW.csv')

basins = flow_table_df["BASIN"].tolist()
flows_to = flow_table_df["FLOWS_TO"].tolist()

# DICTIONARIES
flows_to_dictionary = {basins[k] : flows_to[k] for k, basin in enumerate(basins)}
index_dictionary = {basins[k] : [k] for k, basin in enumerate(basins)}

# Initialize empty basin x basin identity matrix
connectivity_matrix = numpy.identity(numpy.size(basins), dtype = int)
    
######################################### Need to specify the outlet  !!!!!!!!!!!!!!!!!!!!

outlet = "H"

for k, basin in enumerate(basins):
    while basin != outlet:
        connectivity_matrix[k][index_dictionary[flows_to_dictionary[basin]]] = 1
        basin = flows_to_dictionary[basin]

cm_df = pd.DataFrame(connectivity_matrix)
cm_df.index = basins
cm_df.index.name = "BASIN"
cm_df.columns = basins

cm_df.to_csv("input/basin_connectivity_matrix.csv")
# cm_df.to_csv("input_MS_HW/basin_connectivity_matrix_MS_HW.csv")

















