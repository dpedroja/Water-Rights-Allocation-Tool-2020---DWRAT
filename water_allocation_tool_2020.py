# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:29:08 2020

@author: dpedroja
"""
################################# REQURIES NUMPY VERSION 18.1.1 #################################### 

import pulp as pulp
import numpy as np
import pandas as pd
np.__version__

# RAW DATA
# flow
flow_table_df = pd.read_csv('input/flows.csv', index_col = "BASIN")

# demand
rip_demand_df = pd.read_csv('input/riparian_demand.csv')
rip_users = np.array(rip_demand_df["USER"])
app_demand_df = pd.read_csv('input/appropriative_demand.csv')
app_users = np.array(app_demand_df["USER"])
app_users_list = app_users.tolist()

# basically just user location
riparian_basin_user_matrix = np.array(pd.read_csv("input/riparian_user_matrix.csv", index_col="BASIN"))
appropriative_basin_user_matrix = np.array(pd.read_csv("input/appropriative_user_matrix.csv", index_col="BASIN"))

# user connectivity
riparian_user_connectivity_matrix = np.array(pd.read_csv('input/riparian_user_connectivity_matrix.csv', index_col="BASIN"))
appropriative_user_connectivity_matrix = np.array(pd.read_csv('input/appropriative_user_connectivity_matrix.csv', index_col="BASIN"))

# basin connectivity
downstream_connectivity_df = pd.read_csv("input/basin_connectivity_matrix.csv", index_col="BASIN")
downstream_connectivity_df.index.astype(str, copy = False)
downstream_connectivity_matrix = np.array(downstream_connectivity_df)

basins = downstream_connectivity_df.columns.tolist()
upstream_connectivity_matrix = np.transpose(downstream_connectivity_matrix)

# date range for evalutaion
day_range = pd.read_csv("input/day_range.csv")

############################################################################################
# Riparian output files
output_cols = day_range["Dates"].tolist()

rip_basin_proportions_output = pd.DataFrame(columns=[output_cols], index=basins)
# rip_basin_proportions_output.index.name = "BASIN"

# Appropriative output files
app_user_allocations_output = pd.DataFrame(columns=[output_cols], index=app_users_list)
# app_user_allocations_output.index.name = "USER"

############################################################################################

for c, day in enumerate(day_range["Dates"]):
    riparian_demand_data = np.array(rip_demand_df[day])
    net_flow = np.array(flow_table_df[day])
    
    # AVAILABLE FLOW: 
    # available flow: initialize and populate a 1 x k list for available basin flow
    # available flow in basin k is sum of all upstream basin inflows less environmental requirement
    # Matrix operations:
    # upstream connectivity matrix * net flow
    # (k x k) * (k x 1) = k x 1 list of available net basin flows
         #      [1, 0, 0, 0, 0, 0, 0, 0]		[5.6]			[ 5.6]
         #   	[0, 1, 0, 0, 0, 0, 0, 0]		[5.6]			[ 5.6]
         #   	[1, 1, 1, 0, 0, 0, 0, 0]		[5.6]			[16.8]
         #   	[0, 0, 0, 1, 0, 0, 0, 0]	∙	[5.6]		=	[ 5.6]	
         #   	[1, 1, 1, 1, 1, 0, 0, 0]		[5.6]			[28.0]
         #   	[1, 1, 1, 1, 1, 1, 0, 0]		[5.6]			[33.6]
         #   	[0, 0, 0, 0, 0, 0, 1, 0]		[5.6]			[ 5.6]
         #   	[1, 1, 1, 1, 1, 1, 1, 1]		[5.6]			[44.8]
    available_flow_data = np.matmul(upstream_connectivity_matrix, np.transpose(net_flow)) 
    
    # DOWNSTREAM PENALTY
    # number of users upstream of i divided by total users
    # Matrix/vector operations
    # row sum of the k x i user connectivity matrix / count of  i = k x 1 list of downstream penalties
    #           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]				
    #          	[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]				
    #         	[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]				
    #         	[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]	row sum 	=	[ 2,  1,  4,  1,  7,  9,  1, 11]	/	11						
    #        	[0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]				
    #        	[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]				
    #        	[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]				
    #        	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #
    # 	        =	[0.18181818, 0.09090909, 0.36363636, 0.09090909, 0.63636364, 0.81818182, 0.09090909, 1.]
    downstream_penalty_list = np.divide(np.sum(riparian_user_connectivity_matrix, 1), np.count_nonzero(rip_users))
    
    # UPSTREAM BASIN DEMAND
    # basin-wide demand is the sum of user demand upstream of each basin
    # Matrix/vector operations:
    # 1 x i list of user demand ∙ i x k user connectivity matrix  = 1 x k basin demand matrix
    basin_rip_demand_data_T = np.matmul(riparian_demand_data, np.transpose(riparian_user_connectivity_matrix))
   
    # ALPHA
    # minimum of the ratios of downstream penalties to basin demands, element by element division, division by zero should return 0
    alpha = min(np.divide(downstream_penalty_list, basin_rip_demand_data_T, out = np.full_like(downstream_penalty_list, 999999999), where=basin_rip_demand_data_T!=0))

    # ALPHA
    # minimum of the ratios of downstream penalties to basin demands
    # element by element division
    # just a function to take care of any division by zero
    # def safe_min(x, y):
    #     if y.any() <= 0:
    #         return 1000000000000000
    #     return min(x / y)
    # alpha = safe_min(downstream_penalty_list,basin_rip_demand_data_T)

    # DICTIONARIES FOR CONSTRAINTS
    available_flow = {basins[k] : available_flow_data[k] for k, basin in enumerate(basins)}
    downstream_penalty = {basins[k] : downstream_penalty_list[k] for k, basin in enumerate(basins)}
    
    # DEFINE PROBLEM
    Riparian_LP = pulp.LpProblem("RiparianAllocation", pulp.LpMinimize)
    
    # DEFINE DECISION VARIABLES
    basin_proportions = pulp.LpVariable.dicts("Proportions", basins, 0, 1, cat="Continuous")
    # convert dictionary of decision variables to an array
    basin_proportions_list = pd.Series(basin_proportions).values
   
    # USER ALLOCATION
    # user allocation i is their basin's allocation * user i's demand
    # need a 1 x k array of basin proportions ∙ k x i basin user matrix * demand (element-wise) 
    # Matrix/vector operations:
    user_allocation_list = np.multiply((np.matmul(np.transpose(basin_proportions_list), riparian_basin_user_matrix)), riparian_demand_data)
    # dictionary
    user_allocation = {rip_users[i] : user_allocation_list[i] for i, user in enumerate(rip_users)}
    
    # UPSTREAM ALLOCATION: 
    # Sum of user allocations upstream of user i
    # Matrix/vector operations:
    # need k x i upstream user matrix ∙ i by 1 user allocation matrix = k x 1 upstream basin allocation
    #							                    [UserAllocation_A1]	       
    #   	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A2]	    [18*Proportions_A + 0]
    #       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]		[UserAllocation_A3]		[8*Proportions_B + 0]
    #       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A4]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 0]
    #       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]	∙	[UserAllocation_A5]	=	[3*Proportions_D + 0]
    #       [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]		[UserAllocation_A6]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 3*Proportions_D + 13*Proportions_E + 0]
    #       [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]		[UserAllocation_A7]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 3*Proportions_D + 13*Proportions_E + 14*Proportions_F + 0]
    #       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]		[UserAllocation_A8]		[9*Proportions_G + 0]
    #       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]		[UserAllocation_A9]		[18*Proportions_A + 8*Proportions_B + 4*Proportions_C + 3*Proportions_D + 13*Proportions_E + 14*Proportions_F + 9*Proportions_G + 8*Proportions_H + 0]
    #                   							[UserAllocation_A10]
    #                   							[UserAllocation_A11]
    upstream_allocation_list = np.matmul(riparian_user_connectivity_matrix, user_allocation_list)
    # dictionary
    upstream_allocation = {basins[k] : upstream_allocation_list[k] for k, basin in enumerate(basins)}
    
    # OBJECTIVE FUNCTION
    Riparian_LP += alpha * pulp.lpSum([basin_proportions[k]*downstream_penalty[k] for k in basins]) - pulp.lpSum([user_allocation[i] for i in rip_users])
    
    # CONSTRAINTS
    # mass balance
    for k in basins:
        Riparian_LP += pulp.lpSum([upstream_allocation[k]]) <= available_flow[k]
    
    # upstream cannot exceed downstream
    # need k by i downstream proportions matrix
    for k in basins:
        downstream_basins = downstream_connectivity_df.index.astype(str, copy = False)[downstream_connectivity_df[k]==1].tolist()
        for j in downstream_basins:
            Riparian_LP += basin_proportions[j] <= basin_proportions[k]
        
    # SOLVE USING PULP SOLVER
    Riparian_LP.solve()
    print("Status: ", pulp.LpStatus[Riparian_LP.status])
    for v in Riparian_LP.variables():
          print(v.name, "=", v.varValue)
    print("Objective = ", pulp.value(Riparian_LP.objective))
    
    # basin demand dictionary
    basin_demand = {basins[k] : (np.matmul(riparian_basin_user_matrix, riparian_demand_data)[k]) for k,basin in enumerate(basins)}
    
    # this loop is necessary to turn LP output into values
    basin_allocation = []
    for k, basin in enumerate(basins):
        basin_allocation.append(basin_proportions[basin].value() * basin_demand[basin])
     
    print("Basin Total Allocations", basin_allocation)
    # build output table        
    for k in basins:
        rip_basin_proportions_output.loc[k, [day]] = basin_proportions[k].varValue
        
    #############################       APPROPRIATIVE_LP   #######################################################
       
    # UNALLOCATED AVAILABLE FLOW
    # Matrix/vector operations
    # (k x k) upstream connectivity matrix ∙ the k by 1 vector of unallocated flows:
    # (same as riparian in the example, where riparian demand is 0 and appropriative demand = riparian
    #
    #   	[1, 0, 0, 0, 0, 0, 0, 0]		[5.6]			[ 5.6]
    #      	[0, 1, 0, 0, 0, 0, 0, 0]		[5.6]			[ 5.6]
    #      	[1, 1, 1, 0, 0, 0, 0, 0]		[5.6]			[16.8]
    #      	[0, 0, 0, 1, 0, 0, 0, 0]	∙	[5.6]		=	[ 5.6]	
    #      	[1, 1, 1, 1, 1, 0, 0, 0]		[5.6]			[28.0]
    #      	[1, 1, 1, 1, 1, 1, 0, 0]		[5.6]			[33.6]
    #      	[0, 0, 0, 0, 0, 0, 1, 0]		[5.6]			[ 5.6]
    #      	[1, 1, 1, 1, 1, 1, 1, 1]		[5.6]			[44.8]
    
    unallocated_flow = np.array(net_flow - basin_allocation)
    
    if np.sum(np.array(unallocated_flow)) > 0:
        app_available_flow = np.matmul(upstream_connectivity_matrix,unallocated_flow)
        print("Excess flow available for Appropriative allocation:" ,app_available_flow.round(3))
    else:
        app_available_flow = 0
        print("No flow is available for appropriative allocations")
        for i in app_users:
            app_user_allocations_output.loc[i, [day]] = 0
        print(c+1, "of", len(day_range["Dates"]), "complete. Processing day:", day)
        continue
    
    appropriative_demand_data = np.array(app_demand_df[day])
    priority = np.array(app_demand_df["PRIORITY"])
    shortage_penalty_data = np.array([(np.size(app_users)-priority[i]) for i, user in enumerate(app_users)])    
        
    # DICTIONARIES
    app_demand = {app_users[i] : appropriative_demand_data[i] for i, user in enumerate(app_users)}
    shortage_penalty = {app_users[i] : shortage_penalty_data[i] for i, user in enumerate(app_users)}
    app_available_flow = {basins[k] : available_flow_data[k] for k, basin in enumerate(basins)}
    
    # DEFINE PROBLEM
    Appropriative_LP = pulp.LpProblem("AppropriativeProblem", pulp.LpMinimize)
    
    # DEFINE DECISION VARIABLES
    user_allocation = pulp.LpVariable.dicts("UserAllocation", app_users, 0)
    # convert dictionary of decision variables to an array
    user_allocation_list = pd.Series(user_allocation).values
    
    # OBJECTIVE FUNCTION
    Appropriative_LP += pulp.lpSum(  (shortage_penalty[user])*(app_demand[user]-user_allocation[user]) for user in app_users)
    
    # UPSTREAM APPROPRIATIVE BASIN ALLOCATION
    # sum of appropriative allocations upstream of basin k
    # Matrix/vector operations
    # k by i upstream matrix ∙ i by 1 user_allocation for a k by 1 result constrained to available flow.
    #                   							[UserAllocation_A1 ]		
    #   	[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A2 ]		[UserAllocation_A11 + UserAllocation_A4]	     
    #       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]		[UserAllocation_A3 ]		[UserAllocation_A3]
    #       [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]		[UserAllocation_A4 ]		[UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4]
    #       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]	∙	[UserAllocation_A5 ]	=	[UserAllocation_A7]
    #       [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1]		[UserAllocation_A6 ]		[UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4 + UserAllocation_A6 + UserAllocation_A7 + UserAllocation_A8]
    #       [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]		[UserAllocation_A7 ]		[UserAllocation_A1 + UserAllocation_A10 + UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4 + UserAllocation_A6 + UserAllocation_A7 + UserAllocation_A8]
    #       [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]		[UserAllocation_A8 ]		[UserAllocation_A9]
    #       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]		[UserAllocation_A9 ]		[UserAllocation_A1 + UserAllocation_A10 + UserAllocation_A11 + UserAllocation_A2 + UserAllocation_A3 + UserAllocation_A4 + UserAllocation_A5 + UserAllocation_A6 + UserAllocation_A7 + UserAllocation_A8 + UserAllocation_A9]
    #                   							[UserAllocation_A10]
    #                   							[UserAllocation_A11]
    upstream_basin_allocation = np.matmul(appropriative_user_connectivity_matrix, user_allocation_list)
    # dictionary
    upstream_dict = {basins[k] : upstream_basin_allocation[k] for k, basin in enumerate(basins)}
    
    # CONSTRAINTS:
    # 1.  allocation is <= available flow;
    for basin in basins:
        Appropriative_LP += pulp.lpSum(upstream_dict[basin]) <= available_flow[basin]
    # 2.  allocation is <= to reported demand
    for user in app_users:
        Appropriative_LP += pulp.lpSum(user_allocation[user]) <= (app_demand[user])
    # 3. ADDED: sum of user allocation <= total unallocated flow
    Appropriative_LP += pulp.lpSum(user_allocation[i] for i in app_users) <= unallocated_flow
    
    # SOLVE USING PULP SOLVER
    Appropriative_LP.solve()
    print("status:", pulp.LpStatus[Appropriative_LP.status])
    for v in Appropriative_LP.variables():
        print(v.name, "=", v.varValue)
    print("Objective = ", pulp.value(Appropriative_LP.objective))
    
    # this loop is necessary to turn LP output into values
    user_allocations = []
    for i, user in enumerate(app_users):
        user_allocations.append(user_allocation[user].value())

    app_basin_allocations = np.matmul(appropriative_basin_user_matrix, user_allocations)
    print("Basin Appropriative Allocations:") 
    print(app_basin_allocations)
    
    # build output table1
    for i in app_users:
        app_user_allocations_output.loc[i, [day]] = user_allocation[i].varValue
    print(c+1, "of", len(day_range["Dates"]), "complete. Processing day:", day)
    
print("Hi. I'm done")

# app_user_allocations_output.to_csv("C:/Users/dp/sample_app_output.csv", index = True)
# rip_basin_proportions_output.to_csv("C:/Users/dp/sample_rip_output.csv", index = True)
