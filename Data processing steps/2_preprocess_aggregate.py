# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 13:38:29 2021

@author: z0049unj
"""
# %% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import medfilt
from datetime import datetime
import seaborn as sns
from time import perf_counter#to check how long computation takes



import sys
sys.path.append(r'C:\UserData\z0049unj\Documents\Afstuderen\python')

# Import functions 
from preprocess_functions import rows_with_nan
from preprocess_functions import delete_columns
from preprocess_functions import aggregate 
#Define Colors to be used
color_tu=(0, 166/255, 214/255)
plt.rcParams["font.family"] = "sans-serif"
b1=(1/255, 61/255, 168/255)
b2=(41/255, 79/255, 178/255)
b3=(64/255, 97/255, 188/255)
b4=(86/255, 115/255, 197/255)
b5=(106/255, 134/255, 206/255)


b6=(127/255, 152/255, 214/255)
b7=(149/255, 171/255, 222/255)
b8=(171/255, 190/255, 229/255)
b9=(194/255, 208/255, 236/255)
b10=(218/255, 227/255, 243/255)

o1=(242/255, 98/255, 0/255)
o2=(248/255, 115/255, 40/255)
o3=(253/255, 131/255, 65/255)
o4=(255/255, 147/255, 89/255)
o5=(255/255, 163/255, 112/255)
o6=(255/255, 178/255, 135/255)
o7=(255/255, 193/255, 158/255)
o8=(255/255, 209/255, 182/255)
o9=(255/255, 224/255, 206/255)
o10=(255/255, 240/255, 230/255)

locatie_img2=mpimg.imread(r'C:\UserData\z0049unj\Documents\Afstuderen\Data\sensor_locations.png')
plt.imshow(locatie_img2)
plt.title("Location information")

#%% IMPORT PREPROCESSED DATA

loc_501_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_501_F6C.csv')#,index_col=0)
loc_501_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_501_F18C.csv')#,index_col=0)
loc_501r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_501r_F6C.csv')#,index_col=0)
loc_501r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_501r_F18C.csv')#,index_col=0)

loc_502_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_502_F6C.csv')#,index_col=0)
loc_502r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_502r_F6C.csv')#,index_col=0)

loc_528_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_528_F6C.csv')#,index_col=0)
loc_528_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_528_F18C.csv')#,index_col=0)
loc_528r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_528r_F6C.csv')#,index_col=0)
loc_528r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_528r_F18C.csv')#,index_col=0)

loc_530_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_530_F6C.csv')#,index_col=0)
loc_530_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_530_F18C.csv')#,index_col=0)
loc_530r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_530r_F6C.csv')#,index_col=0)
loc_530r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_530r_F18C.csv')#,index_col=0)

loc_531_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_531_F6C.csv')#,index_col=0)
loc_531_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_531_F18C.csv')#,index_col=0)
loc_531r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_531r_F6C.csv')#,index_col=0)
loc_531r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_preprocessed/loc_531r_F18C.csv')#,index_col=0)

# %% Aggregate data into 1 hour frame
#       A. to keep: start_datum, start_tijd, weekday
#       B. Sum:  waarnemingen_intensiteit
#           --> if no missing values just sum
#           --> if x missing values, multiply the sum with 1.x

#   1. Delete unneccessary columns
delete_columns(loc_501_F6C)
delete_columns(loc_501_F18C)  
delete_columns(loc_501r_F6C)
delete_columns(loc_501r_F18C)  

delete_columns(loc_502_F6C)
delete_columns(loc_502r_F6C)

delete_columns(loc_528_F6C)
delete_columns(loc_528_F18C)  
delete_columns(loc_528r_F6C)
delete_columns(loc_528r_F18C)  

delete_columns(loc_530_F6C)
delete_columns(loc_530_F18C)  
delete_columns(loc_530r_F6C)
delete_columns(loc_530r_F18C)  

delete_columns(loc_531_F6C)
delete_columns(loc_531_F18C)  
delete_columns(loc_531r_F6C)
delete_columns(loc_531r_F18C)  

#   2. Obtain rows which still have nan values
[rows_with_nan_501_6,rows_with_nan_501_18,rows_with_nan_501r_6,rows_with_nan_501r_18 ,rows_with_nan_502_6,rows_with_nan_502r_6,rows_with_nan_528_6,rows_with_nan_528_18,rows_with_nan_528r_6,rows_with_nan_528r_18,rows_with_nan_530_6,rows_with_nan_530_18,rows_with_nan_530r_6,rows_with_nan_530r_18,rows_with_nan_531_6,rows_with_nan_531_18,rows_with_nan_531r_6,rows_with_nan_531r_18]=rows_with_nan(loc_501_F6C,loc_501_F18C,loc_501r_F6C,loc_501r_F18C,loc_502_F6C,loc_502r_F6C,loc_528_F6C,loc_528_F18C,loc_528r_F6C,loc_528r_F18C,loc_530_F6C,loc_530_F18C,loc_530r_F6C,loc_530r_F18C,loc_531_F6C,loc_531_F18C,loc_531r_F6C,loc_531r_F18C)


#   3. Aggregate the data
loc_501_F6C=aggregate(loc_501_F6C,rows_with_nan_501_6)
loc_501_F18C=aggregate(loc_501_F18C,rows_with_nan_501_18)
loc_501r_F6C=aggregate(loc_501r_F6C,rows_with_nan_501r_6)
loc_501r_F18C=aggregate(loc_501r_F18C,rows_with_nan_501r_18)

loc_502_F6C=aggregate(loc_502_F6C,rows_with_nan_502_6)
loc_502r_F6C=aggregate(loc_502r_F6C,rows_with_nan_502r_6)

loc_528_F6C=aggregate(loc_528_F6C,rows_with_nan_528_6)
loc_528_F18C=aggregate(loc_528_F18C,rows_with_nan_528_18)
loc_528r_F6C=aggregate(loc_528r_F6C,rows_with_nan_528r_6)
loc_528r_F18C=aggregate(loc_528r_F18C,rows_with_nan_528r_18)

loc_530_F6C=aggregate(loc_530_F6C,rows_with_nan_530_6)
loc_530_F18C=aggregate(loc_530_F18C,rows_with_nan_530_18)
loc_530r_F6C=aggregate(loc_530r_F6C,rows_with_nan_530r_6)
loc_530r_F18C=aggregate(loc_530r_F18C,rows_with_nan_530r_18)

loc_531_F6C=aggregate(loc_531_F6C,rows_with_nan_531_6)
loc_531_F18C=aggregate(loc_531_F18C,rows_with_nan_531_18)
loc_531r_F6C=aggregate(loc_531r_F6C,rows_with_nan_531r_6)
loc_531r_F18C=aggregate(loc_531r_F18C,rows_with_nan_531r_18)


# %% Save all data
#COMMENTED OUT SUCH THAT DOES NOT HAPPEN ON ACCIDENT
'''
loc_501_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501_F6C.csv')
loc_501_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501_F18C.csv')
loc_501r_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501r_F6C.csv')
loc_501r_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501r_F18C.csv')
loc_502_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_502_F6C.csv')
loc_502r_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_502r_F6C.csv')
loc_528_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528_F6C.csv')
loc_528_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528_F18C.csv')
loc_528r_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528r_F6C.csv')
loc_528r_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528r_F18C.csv')
loc_530_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530_F6C.csv')
loc_530_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530_F18C.csv')
loc_530r_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530r_F6C.csv')
loc_530r_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530r_F18C.csv')
loc_531_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531_F6C.csv')
loc_531_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531_F18C.csv')
loc_531r_F6C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531r_F6C.csv')
loc_531r_F18C.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531r_F18C.csv')

'''