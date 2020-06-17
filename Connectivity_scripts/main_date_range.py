# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:33:49 2020

@author: dp
"""

import pandas as pd
import numpy as np
from date_range_fun import date_string


# call function date_string and provide a start and end date in format ("mm/dd/yyy" , "mm/dd/yyy" )
# assign the results of the function to a variable
# 
# output = date_string("4/01/1979", "4/01/1979")



output = date_string("10/02/1995", "10/02/1995")


output = date_string("10/02/1995", "10/12/1995")

# output.to_csv("input_MS_HW/day_range_MS_HW.csv")
output.to_csv("input/day_range.csv")





