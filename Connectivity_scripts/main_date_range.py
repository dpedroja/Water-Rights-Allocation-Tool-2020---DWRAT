# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:33:49 2020

@author: dp
"""
import pandas as pd
import numpy as np
import os

# define a function date_string and provide a start and end date in format ("mm/dd/yyy" , "mm/dd/yyy" )

def convert_date_standard_string(datelike_object):
    """
    Return string version of date in format mm/dd/yyyy
    Parameters
    -----------
    datelike_object
        A value of type date, datetime, or Timestamp.
        (e.g., Python datetime.datetime, datetime.date,
        Pandas Timestamp)
    """
    return "{:%#m/%#d/%Y}".format(datelike_object)
     
def date_string(first, last):
    dates_list = pd.date_range(start=first, end=last)
    dates_df = pd.DataFrame(dates_list, columns=["Dates_list"])
    dates_df["Dates"] = dates_df["Dates_list"].apply(convert_date_standard_string)
    return dates_df

# assign the results of the function to a variable
    
# output = date_string("4/01/1979", "4/01/1979")

output = date_string("10/01/2015", "9/30/2016")

# output.to_csv("input_MS_HW/day_range_MS_HW.csv")


os.getcwd()
os.chdir("C:\\Users\\dp\\WAT_2020_GIT")

output.to_csv("input/day_range.csv")





