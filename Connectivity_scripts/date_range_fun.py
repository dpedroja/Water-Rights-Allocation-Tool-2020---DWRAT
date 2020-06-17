# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 08:59:51 2020

@author: dp
"""
import pandas as pd

# df = pd.read_csv("WEAP_inputs/Daily_Flows.csv")

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
    

# result = date_string("10/2/1999", "10/29/1999")


####################################################################

# result["Dates"].loc[0]
# df["10/2/1999"]
# df[result["Dates"].loc[0]]    
    
    
    
    
    
    
    
    
    
    
    
    
    






