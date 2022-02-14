# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:58:06 2021

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
from preprocess_functions import add_weekday
from preprocess_functions import separate_data_on_location_2019
from preprocess_functions import separate_data_on_location_2018_2017
from preprocess_functions import extract_2lane_sensor_data
from preprocess_functions import extract_502_sensor_data
from preprocess_functions import rows_with_nan
from preprocess_functions import preprocess_interpoleer
from preprocess_functions import preprocess_missing_vel
from preprocess_functions import plot_median_vel
from preprocess_functions import filter_outlier_int_theoretical
from preprocess_functions import filter_outlier_int_kernel
from preprocess_functions import add_kernel
from preprocess_functions import remove_outliers
from preprocess_functions import filter_outlier_velocity

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

siemens_groen=(0/255,153/255,153/255)
siemens_groen_light1=(92/255,176/255,175/255)
siemens_groen_light2=(142/255,198/255,197/255)
siemens_groen_light3=(188/255,221/255,220/255)
siemens_blauw=(1/255,8/255,46/255)

y_green=(72/255,236/255,147/255)
y_blue_d=(30/255,46/255,217/255)
y_blue_l=(157/255,187/255,255/255)
locatie_img2=mpimg.imread(r'C:\UserData\z0049unj\Documents\Afstuderen\Data\sensor_locations.png')
plt.imshow(locatie_img2)
plt.title("Location information")

# %% 1. IMPORT DATA AND MODIFY STRUCTURE
#       A. Import monthly data 2019
data_imported_jan_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-januari/intensiteit-snelheid-export.csv',sep=";")
data_imported_feb_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-februari/intensiteit-snelheid-export.csv',sep=";")
data_imported_maart_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-maart/intensiteit-snelheid-export.csv',sep=";")
data_imported_april_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-april/intensiteit-snelheid-export.csv',sep=";")
data_imported_mei_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-mei/intensiteit-snelheid-export.csv',sep=";")
data_imported_juni_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-juni/intensiteit-snelheid-export.csv',sep=";")
data_imported_juli_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-juli/intensiteit-snelheid-export.csv',sep=";")
data_imported_aug_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-augustus/intensiteit-snelheid-export.csv',sep=";")
data_imported_sep_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-sept/intensiteit-snelheid-export.csv',sep=";")
data_imported_okt_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-okt/intensiteit-snelheid-export.csv',sep=";")
data_imported_nov_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-nov/intensiteit-snelheid-export.csv',sep=";")
data_imported_dec_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/intensiteit-snelheid-export-dec/intensiteit-snelheid-export.csv',sep=";")

print('2019 done')
#       A. Import montly data 2018
data_imported_jan_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-jan-2018/intensiteit-snelheid-export.csv',sep=";")
data_imported_feb_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-feb-2018/intensiteit-snelheid-export.csv',sep=";")
data_imported_maart_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-maart-2018/intensiteit-snelheid-export.csv',sep=";")
data_imported_april_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-april2018/intensiteit-snelheid-export.csv',sep=";")
data_imported_mei_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-mei2018/intensiteit-snelheid-export.csv',sep=";")
data_imported_juni_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-juni2018/intensiteit-snelheid-export-juni.csv',sep=";")
data_imported_juli_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-juli-2018/intensiteit-snelheid-export-juli.csv',sep=";")
data_imported_aug_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-aug-2018/intensiteit-snelheid-export-aug.csv',sep=";")
data_imported_sep_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-sep-2018/intensiteit-snelheid-export.csv',sep=";")
data_imported_okt_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-okt-2018/intensiteit-snelheid-export-okt.csv',sep=";")
data_imported_nov_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-nov-2018/intensiteit-snelheid-export-nov.csv',sep=";")
data_imported_dec_2018=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2018/intensiteit-snelheid-export-dec2018/intensiteit-snelheid-export-dec.csv',sep=";")
print('2018 done')


#       A. Import montly data 2017
data_imported_jan_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-jan-2017/intensiteit-snelheid-export-jan.csv',sep=";")
data_imported_feb_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-feb-2017/intensiteit-snelheid-export-feb.csv',sep=";")
data_imported_maart_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-maart-2017/intensiteit-snelheid-export-mar.csv',sep=";")
data_imported_april_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-april2017/intensiteit-snelheid-export-apr.csv',sep=";")
data_imported_mei_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-mei2017/intensiteit-snelheid-export-mei.csv',sep=";")
data_imported_juni_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-juni2017/intensiteit-snelheid-export-juni.csv',sep=";")
data_imported_juli_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-juli-2017/intensiteit-snelheid-export-juli.csv',sep=";")
data_imported_aug_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-aug-2017/intensiteit-snelheid-export-aug.csv',sep=";")
data_imported_sep_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-sep-2017/intensiteit-snelheid-export-sep.csv',sep=";")
data_imported_okt_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-okt-2017/intensiteit-snelheid-export-okt.csv',sep=";")
data_imported_nov_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-nov-2017/intensiteit-snelheid-export-nov.csv',sep=";")
data_imported_dec_2017=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/2017/intensiteit-snelheid-export-dec2017/intensiteit-snelheid-export.csv',sep=";")
print('2017 done')

#    B. Add weekday as a column
add_weekday(data_imported_jan_2019,data_imported_feb_2019,data_imported_maart_2019,data_imported_april_2019,data_imported_mei_2019,data_imported_juni_2019,data_imported_juli_2019,data_imported_aug_2019,data_imported_sep_2019,data_imported_okt_2019,data_imported_nov_2019,data_imported_dec_2019)
add_weekday(data_imported_jan_2018,data_imported_feb_2018,data_imported_maart_2018,data_imported_april_2018,data_imported_mei_2018,data_imported_juni_2018,data_imported_juli_2018,data_imported_aug_2018,data_imported_sep_2018,data_imported_okt_2018,data_imported_nov_2018,data_imported_dec_2018)
add_weekday(data_imported_jan_2017,data_imported_feb_2017,data_imported_maart_2017,data_imported_april_2017,data_imported_mei_2017,data_imported_juni_2017,data_imported_juli_2017,data_imported_aug_2017,data_imported_sep_2017,data_imported_okt_2017,data_imported_nov_2017,data_imported_dec_2017)

#       C. Separate data based on location

yearly_data_2019=[data_imported_jan_2019,data_imported_feb_2019,data_imported_maart_2019,data_imported_april_2019,data_imported_mei_2019,data_imported_juni_2019,data_imported_juli_2019,data_imported_aug_2019,data_imported_sep_2019,data_imported_okt_2019,data_imported_nov_2019,data_imported_dec_2019]
yearly_data_2018=[data_imported_jan_2018,data_imported_feb_2018,data_imported_maart_2018,data_imported_april_2018,data_imported_mei_2018,data_imported_juni_2018,data_imported_juli_2018,data_imported_aug_2018,data_imported_sep_2018,data_imported_okt_2018,data_imported_nov_2018,data_imported_dec_2018]
yearly_data_2017=[data_imported_jan_2017,data_imported_feb_2017,data_imported_maart_2017,data_imported_april_2017,data_imported_mei_2017,data_imported_juni_2017,data_imported_juli_2017,data_imported_aug_2017,data_imported_sep_2017,data_imported_okt_2017,data_imported_nov_2017,data_imported_dec_2017]

def seperate_data_on_location_2019(Data_yearly):
    for i in range(12): 
        if i == 0:
            [loc_501,loc_501r,loc_502,loc_502r,loc_528,loc_528r,loc_530,loc_530r,loc_531,loc_531r]=separate_data_on_location_2019("PNH02_PNHTI501",'PNH02_PNHTI501r','PNH02_PNHTI502','PNH02_PNHTI502r','PNH02_PNHTI528','PNH02_PNHTI528r','PNH02_PNHTI530','PNH02_PNHTI530r','PNH02_PNHTI531','PNH02_PNHTI531r',Data_yearly[i])
        else:
            [loc_501_month,loc_501r_month,loc_502_month,loc_502r_month,loc_528_month,loc_528r_month,loc_530_month,loc_530r_month,loc_531_month,loc_531r_month]=separate_data_on_location_2019("PNH02_PNHTI501",'PNH02_PNHTI501r','PNH02_PNHTI502','PNH02_PNHTI502r','PNH02_PNHTI528','PNH02_PNHTI528r','PNH02_PNHTI530','PNH02_PNHTI530r','PNH02_PNHTI531','PNH02_PNHTI531r',Data_yearly[i])
            loc_501=loc_501.append(loc_501_month,ignore_index=True)
            loc_501r=loc_501r.append(loc_501r_month,ignore_index=True)
            loc_502=loc_502.append(loc_502_month,ignore_index=True)
            loc_502r=loc_502r.append(loc_502r_month,ignore_index=True)
            loc_528=loc_528.append(loc_528_month,ignore_index=True)
            loc_528r=loc_528r.append(loc_528r_month,ignore_index=True)
            loc_530=loc_530.append(loc_530_month,ignore_index=True)
            loc_530r=loc_530r.append(loc_530r_month,ignore_index=True)
            loc_531=loc_531.append(loc_531_month,ignore_index=True)
            loc_531r=loc_531r.append(loc_531r_month,ignore_index=True)
            
    return loc_501,loc_501r,loc_502,loc_502r,loc_528,loc_528r,loc_530,loc_530r,loc_531,loc_531r

def seperate_data_on_location_2018_2017(Data_yearly):
    for i in range(12): 
        if i == 0:
            [loc_501,loc_501r,loc_502,loc_502r,loc_528,loc_528r,loc_530,loc_530r,loc_531,loc_531r]=separate_data_on_location_2018_2017("PNH02_PNHTI501",'PNH02_PNHTI501r','PNH02_PNHTI502','PNH02_PNHTI502r','PNH02_PNHTI528','PNH02_PNHTI528r','PNH02_PNHTI530','PNH02_PNHTI530r','PNH02_PNHTI531','PNH02_PNHTI531r',Data_yearly[i])
        else:
            [loc_501_month,loc_501r_month,loc_502_month,loc_502r_month,loc_528_month,loc_528r_month,loc_530_month,loc_530r_month,loc_531_month,loc_531r_month]=separate_data_on_location_2018_2017("PNH02_PNHTI501",'PNH02_PNHTI501r','PNH02_PNHTI502','PNH02_PNHTI502r','PNH02_PNHTI528','PNH02_PNHTI528r','PNH02_PNHTI530','PNH02_PNHTI530r','PNH02_PNHTI531','PNH02_PNHTI531r',Data_yearly[i])
            loc_501=loc_501.append(loc_501_month,ignore_index=True)
            loc_501r=loc_501r.append(loc_501r_month,ignore_index=True)
            loc_502=loc_502.append(loc_502_month,ignore_index=True)
            loc_502r=loc_502r.append(loc_502r_month,ignore_index=True)
            loc_528=loc_528.append(loc_528_month,ignore_index=True)
            loc_528r=loc_528r.append(loc_528r_month,ignore_index=True)
            loc_530=loc_530.append(loc_530_month,ignore_index=True)
            loc_530r=loc_530r.append(loc_530r_month,ignore_index=True)
            loc_531=loc_531.append(loc_531_month,ignore_index=True)
            loc_531r=loc_531r.append(loc_531r_month,ignore_index=True)
            
    return loc_501,loc_501r,loc_502,loc_502r,loc_528,loc_528r,loc_530,loc_530r,loc_531,loc_531r


[loc_501_2019,loc_501r_2019,loc_502_2019,loc_502r_2019,loc_528_2019,loc_528r_2019,loc_530_2019,loc_530r_2019,loc_531_2019,loc_531r_2019]=seperate_data_on_location_2019(yearly_data_2019)
[loc_501_2018,loc_501r_2018,loc_502_2018,loc_502r_2018,loc_528_2018,loc_528r_2018,loc_530_2018,loc_530r_2018,loc_531_2018,loc_531r_2018]=seperate_data_on_location_2018_2017(yearly_data_2018)
[loc_501_2017,loc_501r_2017,loc_502_2017,loc_502r_2017,loc_528_2017,loc_528r_2017,loc_530_2017,loc_530r_2017,loc_531_2017,loc_531r_2017]=seperate_data_on_location_2018_2017(yearly_data_2017)

#       D. concat different years 2017->2018->2019
loc_501=pd.concat([loc_501_2017,loc_501_2018,loc_501_2019],axis=0)
loc_501r=pd.concat([loc_501r_2017,loc_501r_2018,loc_501r_2019],axis=0)
loc_502=pd.concat([loc_502_2017,loc_502_2018,loc_502_2019],axis=0)
loc_502r=pd.concat([loc_502r_2017,loc_502r_2018,loc_502r_2019],axis=0)
loc_528=pd.concat([loc_528_2017,loc_528_2018,loc_528_2019],axis=0)
loc_528r=pd.concat([loc_528r_2017,loc_528r_2018,loc_528r_2019],axis=0)
loc_530=pd.concat([loc_530_2017,loc_530_2018,loc_530_2019],axis=0)
loc_530r=pd.concat([loc_530r_2017,loc_530r_2018,loc_530r_2019],axis=0)
loc_531=pd.concat([loc_531_2017,loc_531_2018,loc_531_2019],axis=0)
loc_531r=pd.concat([loc_531r_2017,loc_531r_2018,loc_531r_2019],axis=0)

#        E. Remove data from sensors that is not required
    
[loc_501_F6C,loc_501_F18C]=extract_2lane_sensor_data(loc_501)
[loc_501r_F6C,loc_501r_F18C]=extract_2lane_sensor_data(loc_501r)

loc_502_F6C=extract_502_sensor_data(loc_502)
loc_502r_F6C=extract_502_sensor_data(loc_502r)

[loc_528_F6C,loc_528_F18C]=extract_2lane_sensor_data(loc_528)
[loc_528r_F6C,loc_528r_F18C]=extract_2lane_sensor_data(loc_528r)
[loc_530_F6C,loc_530_F18C]=extract_2lane_sensor_data(loc_530)
[loc_530r_F6C,loc_530r_F18C]=extract_2lane_sensor_data(loc_530r)
[loc_531_F6C,loc_531_F18C]=extract_2lane_sensor_data(loc_531)
[loc_531r_F6C,loc_531r_F18C]=extract_2lane_sensor_data(loc_531r)

# %% Plot plaatje voor locatie 502 jan 2018 zodat duidelijk die nul waardes

x_502=loc_502_F6C.loc[(loc_502_F6C['start_datum']>='2018-01-01' )&( loc_502_F6C['start_datum']<'2018-01-3'),'waarnemingen_intensiteit'].reset_index(drop=True)
x_502r=loc_502r_F6C.loc[(loc_502r_F6C['start_datum']>='2018-01-01' )&( loc_502r_F6C['start_datum']<'2018-01-3'),'waarnemingen_intensiteit'].reset_index(drop=True)


fig,ax= plt.subplots()
ax.plot(x_502r,color=siemens_groen_light2,label='Location 502r lane 1')
ax.plot(x_502,color=siemens_groen,label='Location 502 lane 1')


my_xticks=['01/01','01/02','01/03','01/04','01/05','01/06','01/07']
my_xticks = ['00:00:00','03:00:00','06:00:00','09:00:00','12:00:00','15:00:00','18:00:00','21:00:00','00:00:00','03:00:00','06:00:00','09:00:00','12:00:00','15:00:00','18:00:00','21:00:00']

ax.set_xticks(np.linspace(0, 576-576/17,16))
ax.set_xticklabels(my_xticks)
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('Time [hh:mm:ss]')
plt.ylabel('Traffic flow [veh/5min]')
plt.title('Traffic flow at 01-01-2018 and 01-02-2018')
plt.ylim(-2,100,20)


plt.savefig(r"C:\UserData\z0049unj\Documents\Afstuderen\python\Preprocess_without_inserting_median\figurespdf\502_jan_zero_data.pdf",bbox_inches='tight')

# %% 2.     FILTER OUTLIERS INTENSITY
#   A. Check possible intensity at each road type
#      --> if higher than theoretical intensity remove datapoint
#   B. Check if surrounding points of peaks are also high, then possible otherwise probably measurement error
#           1. Look into statistics to find indices of points which should be investigated
#           2. Add sliding window column to the data
#              --> kernel=7, 15 minutes before and 15 minutes after               

#           3. Check conditions with kernel and data
#              --> Remove data if it does not meet the conditions

#       0. determine the maximum velocity, based on median for each loaction.
''' Uncomment if desired to plot the statistics of the velocity for each location
plot_median_vel(loc_501_F6C)
plot_median_vel(loc_501_F18C)
plot_median_vel(loc_501r_F6C)
plot_median_vel(loc_501r_F18C)

plot_median_vel(loc_502_F6C)
plot_median_vel(loc_502r_F6C)

plot_median_vel(loc_528_F6C)
plot_median_vel(loc_528_F18C)
plot_median_vel(loc_528r_F6C)
plot_median_vel(loc_528r_F18C)

plot_median_vel(loc_530_F6C)
plot_median_vel(loc_530_F18C)
plot_median_vel(loc_530r_F6C)
plot_median_vel(loc_530r_F18C)

plot_median_vel(loc_531_F6C)
plot_median_vel(loc_531_F18C)
plot_median_vel(loc_531r_F6C)
plot_median_vel(loc_531r_F18C)
'''

#      1. Determine the maximum intensity
#max_velocity=np.array([80, 50, 70 ,70, 70])+30       #30 km/h  to hard
#max_velocity=np.array([115,100,115,100,65,65,85,85,85,85,105,110,110,90,120,100,120,100])+30    #max van 95% and 30 km/h more to be sure
max_velocity=np.array([80,50,70])*1.3

v_tijd=1.5                                          #volgtijd

totale_afstand=max_velocity*1000/60*5               #m per 5 min
v_afstand=max_velocity/60/60*1000*v_tijd
v_totaal_afstand=v_afstand+4                        #4 m lengte gemiddelde auto
max_intensiteit=totale_afstand/v_totaal_afstand
print(max_intensiteit)

#      2. Filter out the high peaks
#loc_501_F6C_filtered=filter_outlier_int_theoretical(loc_501_F6C.loc[loc_501_F6C.index[loc_501_F6C['start_datum']<'2018-01-01']],max_intensiteit[0],'501','1')
#loc_501_F6C_filtered=filter_outlier_int_theoretical(loc_501_F6C.loc[loc_501_F6C.index[loc_501_F6C['start_datum']>'2018-12-31']],max_intensiteit[0],'501','1')

# %% Make two nice plots
locatie=loc_528r_F18C
data=locatie.loc[locatie.index[locatie['start_datum']>'2018-12-31']].reset_index()
condition=data['waarnemingen_intensiteit']>max_intensiteit[0]
loc_filtered=data.loc[data.index[condition],:]
    

my_xticks=['1','2','3','4','5','6','7','8','9','10','11','12']
#my_xticks=['Jan','Feb','Mar','April','May','June','July','Aug','Sep','Oct','Nov','Dec']
fig,ax= plt.subplots()
ax.plot(data['waarnemingen_intensiteit'],color=siemens_groen_light2,label='Real data')
ax.scatter(loc_filtered.index,loc_filtered['waarnemingen_intensiteit'],color=siemens_blauw,label='Filtered points')

    #change waarnemingen_intensiteit and gem_snelheid at these locations to nan
data.loc[loc_filtered.index,'waarnemingen_intensiteit']=np.nan
data.loc[loc_filtered.index,'gem_snelheid']=np.nan
ax.plot(data['waarnemingen_intensiteit'],color=siemens_groen,label='Filtered data')
    
    
ax.set_xticks(np.linspace(4380, 100740,12))
ax.set_xticklabels(my_xticks)
plt.legend()
plt.xlabel('Month of the year')
plt.ylabel('Traffic flow [veh/5min]')
plt.title("Traffic flow in 2019 at location 528r and lane 2")
plt.ylim((0,920))

plt.savefig(r"C:\UserData\z0049unj\Documents\Afstuderen\python\Preprocess_without_inserting_median\figurespdf\filtered_528r.pdf",bbox_inches='tight')


plt.show()

# %% 
loc_501_F6C_filtered=filter_outlier_int_theoretical(loc_501_F6C,max_intensiteit[0])
loc_501_F18C_filtered=filter_outlier_int_theoretical(loc_501_F18C,max_intensiteit[0])
loc_501r_F6C_filtered=filter_outlier_int_theoretical(loc_501r_F6C,max_intensiteit[0])
loc_501r_F18C_filtered=filter_outlier_int_theoretical(loc_501r_F18C,max_intensiteit[0])

loc_502_F6C_filtered=filter_outlier_int_theoretical(loc_502_F6C,max_intensiteit[1])
loc_502r_F6C_filtered=filter_outlier_int_theoretical(loc_502r_F6C,max_intensiteit[1])

loc_528_F6C_filtered=filter_outlier_int_theoretical(loc_528_F6C,max_intensiteit[2])
loc_528_F18C_filtered=filter_outlier_int_theoretical(loc_528_F18C,max_intensiteit[2])
loc_528r_F6C_filtered=filter_outlier_int_theoretical(loc_528r_F6C,max_intensiteit[2])
loc_528r_F18C_filtered=filter_outlier_int_theoretical(loc_528r_F18C,max_intensiteit[2])

loc_530_F6C_filtered=filter_outlier_int_theoretical(loc_530_F6C,max_intensiteit[2])
loc_530_F18C_filtered=filter_outlier_int_theoretical(loc_530_F18C,max_intensiteit[2])
loc_530r_F6C_filtered=filter_outlier_int_theoretical(loc_530r_F6C,max_intensiteit[2])
loc_530r_F18C_filtered=filter_outlier_int_theoretical(loc_530r_F18C,max_intensiteit[2])

loc_531_F6C_filtered=filter_outlier_int_theoretical(loc_531_F6C,max_intensiteit[2])
loc_531_F18C_filtered=filter_outlier_int_theoretical(loc_531_F18C,max_intensiteit[2])
loc_531r_F6C_filtered=filter_outlier_int_theoretical(loc_531r_F6C,max_intensiteit[2])
loc_531r_F18C_filtered=filter_outlier_int_theoretical(loc_531r_F18C,max_intensiteit[2])

#       3. Print how many values are filtered
print('how many values filtered due to theoretical maximum')
print(     'loc 501 F6C:', loc_501_F6C_filtered.shape[0])           #7
print(      'loc 501 F18C:', loc_501_F18C_filtered.shape[0])        #8
print(      'loc 501r F6C:', loc_501r_F6C_filtered.shape[0])        #1
print(      'loc 501r F18C:', loc_501r_F18C_filtered.shape[0])      #0

print(      'loc 502 F6C:', loc_502_F6C_filtered.shape[0])          #191
print(      'loc 502r F6C:', loc_502r_F6C_filtered.shape[0])        #535
    
print(      'loc 528 F6C:', loc_528_F6C_filtered.shape[0])          #0
print(      'loc 528 18C:', loc_528_F18C_filtered.shape[0])         #0
print(      'loc 528r F6C:', loc_528r_F6C_filtered.shape[0])        #1071
print(      'loc 528r F18C:', loc_528r_F18C_filtered.shape[0])      #1992

print(      'loc 530 F6C:', loc_530_F6C_filtered.shape[0])          #289
print(      'loc 530 F18C:', loc_530_F18C_filtered.shape[0])        #0
print(     'loc 530r F6C:', loc_530r_F6C_filtered.shape[0])         #66
print(      'loc 530r F18C:', loc_530r_F18C_filtered.shape[0])      #0

print(      'loc 531 F6C:', loc_531_F6C_filtered.shape[0])          #2
print(      'loc 531 F18C:', loc_531_F18C_filtered.shape[0])        #782
print(      'loc 531r F6C:', loc_531r_F6C_filtered.shape[0])        #0
print(      'loc 531r F18C:', loc_531r_F18C_filtered.shape[0])      #0

# %%    B. Check if surrounding points of peaks are also high

#   1. Obtain indices of the datapoints which should be checked
    
loc_501_index_total_F6C=filter_outlier_int_kernel(loc_501_F6C)
loc_501_index_total_F18C=filter_outlier_int_kernel(loc_501_F18C)
loc_501r_index_total_F6C=filter_outlier_int_kernel(loc_501r_F6C)
loc_501r_index_total_F18C=filter_outlier_int_kernel(loc_501r_F18C)

loc_502_index_total_F6C=filter_outlier_int_kernel(loc_502_F6C)
loc_502r_index_total_F6C=filter_outlier_int_kernel(loc_502r_F6C)

loc_528_index_total_F6C=filter_outlier_int_kernel(loc_528_F6C)
loc_528_index_total_F18C=filter_outlier_int_kernel(loc_528_F18C)
loc_528r_index_total_F6C=filter_outlier_int_kernel(loc_528r_F6C)
loc_528r_index_total_F18C=filter_outlier_int_kernel(loc_528r_F18C)

loc_530_index_total_F6C=filter_outlier_int_kernel(loc_530_F6C)
loc_530_index_total_F18C=filter_outlier_int_kernel(loc_530_F18C)
loc_530r_index_total_F6C=filter_outlier_int_kernel(loc_530r_F6C)
loc_530r_index_total_F18C=filter_outlier_int_kernel(loc_530r_F18C)

loc_531_index_total_F6C=filter_outlier_int_kernel(loc_531_F6C)
loc_531_index_total_F18C=filter_outlier_int_kernel(loc_531_F18C)
loc_531r_index_total_F6C=filter_outlier_int_kernel(loc_531r_F6C)
loc_531r_index_total_F18C=filter_outlier_int_kernel(loc_531r_F18C)


#   2. Add sliding window column to the data
#       A. Condition1: data 3* kernel?
#       B. Condition2: data>50\
#          --> return the data which should be filtered (data_final).

[loc_501_data_final_F6C,loc_501_data_final_F18C]=add_kernel(loc_501_F6C,loc_501_F18C,7,loc_501_index_total_F6C,loc_501_index_total_F18C)
[loc_501r_data_final_F6C,loc_501r_data_final_F18C]=add_kernel(loc_501r_F6C,loc_501r_F18C,7,loc_501r_index_total_F6C,loc_501r_index_total_F18C)

[loc_502_data_final_F6C,loc_502r_data_final_F6C]=add_kernel(loc_502_F6C,loc_502r_F6C,7,loc_502_index_total_F6C,loc_502r_index_total_F6C)

[loc_528_data_final_F6C,loc_528_data_final_F18C]=add_kernel(loc_528_F6C,loc_528_F18C,7,loc_528_index_total_F6C,loc_528_index_total_F18C)
[loc_528r_data_final_F6C,loc_528r_data_final_F18C]=add_kernel(loc_528r_F6C,loc_528r_F18C,7,loc_528r_index_total_F6C,loc_528r_index_total_F18C)

[loc_530_data_final_F6C,loc_530_data_final_F18C]=add_kernel(loc_530_F6C,loc_530_F18C,7,loc_530_index_total_F6C,loc_530_index_total_F18C)
[loc_530r_data_final_F6C,loc_530r_data_final_F18C]=add_kernel(loc_530r_F6C,loc_530r_F18C,7,loc_530r_index_total_F6C,loc_530r_index_total_F18C)

[loc_531_data_final_F6C,loc_531_data_final_F18C]=add_kernel(loc_531_F6C,loc_531_F18C,7,loc_531_index_total_F6C,loc_531_index_total_F18C)
[loc_531r_data_final_F6C,loc_531r_data_final_F18C]=add_kernel(loc_531r_F6C,loc_531r_F18C,7,loc_531r_index_total_F6C,loc_531r_index_total_F18C)

# %% 3. Remove the outliers from the data
# not removed yet, inserted nan at waarnemingen intensiteit
remove_outliers(loc_501_F6C,loc_501_data_final_F6C) 
remove_outliers(loc_501_F18C,loc_501_data_final_F18C)
remove_outliers(loc_501r_F6C,loc_501r_data_final_F6C)
remove_outliers(loc_501r_F18C,loc_501r_data_final_F18C)

remove_outliers(loc_502_F6C,loc_502_data_final_F6C)
remove_outliers(loc_502r_F6C,loc_502r_data_final_F6C)

remove_outliers(loc_528_F6C,loc_528_data_final_F6C)
remove_outliers(loc_528_F18C,loc_528_data_final_F18C)
remove_outliers(loc_528r_F6C,loc_528r_data_final_F6C)
remove_outliers(loc_528r_F18C,loc_528r_data_final_F18C)

remove_outliers(loc_530_F6C,loc_530_data_final_F6C)
remove_outliers(loc_530_F18C,loc_530_data_final_F18C)
remove_outliers(loc_530r_F6C,loc_530r_data_final_F6C)
remove_outliers(loc_530r_F18C,loc_530r_data_final_F18C)

remove_outliers(loc_531_F6C,loc_531_data_final_F6C)
remove_outliers(loc_531_F18C,loc_531_data_final_F18C)
remove_outliers(loc_531r_F6C,loc_531r_data_final_F6C)
remove_outliers(loc_531r_F18C,loc_531r_data_final_F18C)


#   4. Print how many values are filtered
print('how many values filtered due to kernel') 
print(     'loc 501 F6C:', loc_501_data_final_F6C.shape[0])             # 47
print(      'loc 501 F18C:', loc_501_data_final_F18C.shape[0])          #31
print(      'loc 501r F6C:', loc_501r_data_final_F6C.shape[0])          #4
print(      'loc 501r F18C:', loc_501r_data_final_F18C.shape[0])        #2

print(     'loc 502 F6C:', loc_502_data_final_F6C.shape[0])             #41
print(      'loc 502r F6C:', loc_502r_data_final_F6C.shape[0])          #60

print(     'loc 528 F6C:', loc_528_data_final_F6C.shape[0])             #1
print(      'loc 528 F18C:', loc_528_data_final_F18C.shape[0])          #2
print(      'loc 528r F6C:', loc_528r_data_final_F6C.shape[0])          #381
print(      'loc 528r F18C:', loc_528r_data_final_F18C.shape[0])        #355

print(     'loc 530 F6C:', loc_530_data_final_F6C.shape[0])             #9
print(      'loc 530 F18C:', loc_530_data_final_F18C.shape[0])          #2
print(      'loc 530r F6C:', loc_530r_data_final_F6C.shape[0])          #6
print(      'loc 530r F18C:', loc_530r_data_final_F18C.shape[0])        #0

print(     'loc 531 F6C:', loc_531_data_final_F6C.shape[0])             #17
print(      'loc 531 F18C:', loc_531_data_final_F18C.shape[0])          #118
print(      'loc 531r F6C:', loc_531r_data_final_F6C.shape[0])          #11
print(      'loc 531r F18C:', loc_531r_data_final_F18C.shape[0])        #1

# %% 3. FILTER OUTLIERS VELOCITY
#       --> set to mediaan if too fast, i do not want to model this behavior and sometimes even so high due to measurements errors
#       --> However i do not want to throw away data of intensity, therefore not set to nan

loc_501_F6C_too_fast=filter_outlier_velocity(loc_501_F6C,max_velocity[0])
loc_501_F18C_too_fast=filter_outlier_velocity(loc_501_F18C,max_velocity[0])
loc_501r_F6C_too_fast=filter_outlier_velocity(loc_501r_F6C,max_velocity[0])
loc_501r_F18C_too_fast=filter_outlier_velocity(loc_501r_F18C,max_velocity[0])

loc_502_F6C_too_fast=filter_outlier_velocity(loc_502_F6C,max_velocity[1])
loc_502r_F6C_too_fast=filter_outlier_velocity(loc_502r_F6C,max_velocity[1])

loc_528_F6C_too_fast=filter_outlier_velocity(loc_528_F6C,max_velocity[2])
loc_528_F18C_too_fast=filter_outlier_velocity(loc_528_F18C,max_velocity[2])
loc_528r_F6C_too_fast=filter_outlier_velocity(loc_528r_F6C,max_velocity[2])
loc_528r_F18C_too_fast=filter_outlier_velocity(loc_528r_F18C,max_velocity[2])

loc_530_F6C_too_fast=filter_outlier_velocity(loc_530_F6C,max_velocity[2])
loc_530_F18C_too_fast=filter_outlier_velocity(loc_530_F18C,max_velocity[2])
loc_530r_F6C_too_fast=filter_outlier_velocity(loc_530r_F6C,max_velocity[2])
loc_530r_F18C_too_fast=filter_outlier_velocity(loc_530r_F18C,max_velocity[2])

loc_531_F6C_too_fast=filter_outlier_velocity(loc_531_F6C,max_velocity[2])
loc_531_F18C_too_fast=filter_outlier_velocity(loc_531_F18C,max_velocity[2])
loc_531r_F6C_too_fast=filter_outlier_velocity(loc_531r_F6C,max_velocity[2])
loc_531r_F18C_too_fast=filter_outlier_velocity(loc_531r_F18C,max_velocity[2])


# %% 4. REMOVE NAN DATA

#       A. find out how many NaN each dataframe
all_data=[loc_501_F6C,loc_501_F18C,loc_501r_F6C,loc_501r_F18C,loc_502_F6C,loc_502r_F6C,loc_528_F6C,loc_528_F18C,loc_528r_F6C,loc_528r_F18C,loc_530_F6C,loc_530_F18C,loc_530r_F6C,loc_530r_F18C,loc_531_F6C,loc_531_F18C,loc_531r_F6C,loc_531r_F18C]

for i in range(18):
    print(i, all_data[i].isnull().sum() )

'''
 246 missing for each location and for location 502(r) 245 dit was eerst zonder dat gekke waardes eruit waren gefilterd nu:
     loc501 F6C: 1060
     loc 501 F18C: 1045
     loc 501r F6C: 1011
     loc 501r F18C: 1008
     
     loc 502 F6C: 1216
     loc 502r F6C: 1579
     
     loc 528 F6C: 1023
     loc 528 F18C: 1024
     loc 528r F6C: 2437
     loc 528r F18C: 3332
     
     loc 530 F6C: 1294
     loc 530 F18C: 998
     loc 530r F6C: 1068
     loc 530r F18C: 996
     
     loc 531 F6C: 1011
     loc 531 F18C: 1892
     loc 531r F6C: 1003
     loc 531r F18C: 993
'''     
# %%plot one figure in which it is clear that points are missing

date_501=loc_501_F6C.loc[loc_501_F6C.index[loc_501_F6C['start_datum']=='2019-01-04'],:]
date_501r=loc_501r_F6C.loc[loc_501r_F6C.index[loc_501r_F6C['start_datum']=='2019-01-04'],:]
date_502=loc_502_F6C.loc[loc_502_F6C.index[loc_502_F6C['start_datum']=='2019-01-04'],:]
date_502r=loc_501r_F6C.loc[loc_502r_F6C.index[loc_502r_F6C['start_datum']=='2019-01-04'],:]
date_528=loc_528_F6C.loc[loc_528_F6C.index[loc_528_F6C['start_datum']=='2019-01-04'],:]

x = np.linspace(0,287,288)
my_xticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']



fig,ax= plt.subplots()
ax.plot(x,date_501['waarnemingen_intensiteit'],color=siemens_groen,label='Location 501 lane 1')
#ax.plot(x,date_501r['waarnemingen_intensiteit'],color=siemens_groen_light1,label='Location 501r lane 1')
ax.plot(x,date_502['waarnemingen_intensiteit'],color=siemens_groen_light2,label='Location 502 lane 1')
ax.plot(x,date_528['waarnemingen_intensiteit'],color=siemens_groen_light3,label='Location 528 lane 1')

ax.set_xticks(np.linspace(0,287,24))
ax.set_xticklabels(my_xticks)
plt.xticks(rotation=60)
plt.legend()
plt.xlabel('Time [hh:mm:ss]')
plt.ylabel('Traffic flow [veh/5min]')
plt.title("Traffic flow at 01-04-2019")
plt.ylim(-2,100,20)

plt.savefig(r"C:\UserData\z0049unj\Documents\Afstuderen\python\Preprocess_without_inserting_median\figurespdf\traffic_flow_nan.pdf",bbox_inches='tight')


plt.show()

#%% Plot figures with percentile ranges
my_xticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']

loc_dow=loc_501_F6C.groupby(['weekday'])
monday_data=loc_dow.get_group(6)

#nu door naar statistics boxplot
data_stats = monday_data.groupby(['start_tijd']).describe(percentiles=[.05,.25, .5, .75,.95])
median = data_stats['waarnemingen_intensiteit', '50%']
median.name = 'Traffic flow'
quartiles1 = data_stats['waarnemingen_intensiteit', '25%']
quartiles3 = data_stats['waarnemingen_intensiteit', '75%']
q_5= data_stats['waarnemingen_intensiteit', '5%']
q_95= data_stats['waarnemingen_intensiteit', '95%']
x = data_stats.index
x = np.linspace(0,287,288)

        #UNCOMMENT IF DESIRED TO PLOT
fig,ax =plt.subplots()
ax=sns.lineplot(x=x, y=median,color=siemens_blauw,label='Median')
ax.fill_between(x, q_5, q_95,color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, quartiles1, quartiles3,color=siemens_groen,label='25%-75%')
plt.legend()
plt.title('Statistics location 501 and lane 1 at Sunday') 
ax.set_xticks(np.linspace(0,287,24))
ax.set_xticklabels(my_xticks)
plt.xticks(rotation=80)
plt.xlabel('Time [hh:mm:ss]')
plt.ylabel('Traffic flow [veh/5min]')
plt.ylim((0,50))

plt.savefig("medianstats_sun.svg",bbox_inches='tight')

plt.show()


#%%
#       A. Find rows with nans
[rows_with_nan_501_6,rows_with_nan_501_18,rows_with_nan_501r_6,rows_with_nan_501r_18 ,rows_with_nan_502_6,rows_with_nan_502r_6,rows_with_nan_528_6,rows_with_nan_528_18,rows_with_nan_528r_6,rows_with_nan_528r_18,rows_with_nan_530_6,rows_with_nan_530_18,rows_with_nan_530r_6,rows_with_nan_530r_18,rows_with_nan_531_6,rows_with_nan_531_18,rows_with_nan_531r_6,rows_with_nan_531r_18]=rows_with_nan(loc_501_F6C,loc_501_F18C,loc_501r_F6C,loc_501r_F18C,loc_502_F6C,loc_502r_F6C,loc_528_F6C,loc_528_F18C,loc_528r_F6C,loc_528r_F18C,loc_530_F6C,loc_530_F18C,loc_530r_F6C,loc_530r_F18C,loc_531_F6C,loc_531_F18C,loc_531r_F6C,loc_531r_F18C)


# %% Fix missing values
#   1. if number of consequtive missing values <=3 do not change the data
#   2. if number of consequtive missing values > 3:
#       A. if between 00:00:55 and 04:00:00
#           --> insert median
#       B. if during the day
#           --> remove day from data set
                        
preprocess_interpoleer(loc_501_F6C, rows_with_nan_501_6)
preprocess_interpoleer(loc_501_F18C, rows_with_nan_501_18)
preprocess_interpoleer(loc_501r_F6C, rows_with_nan_501r_6)
preprocess_interpoleer(loc_501r_F18C, rows_with_nan_501r_18)

preprocess_interpoleer(loc_502_F6C, rows_with_nan_502_6)
preprocess_interpoleer(loc_502r_F6C, rows_with_nan_502r_6)

preprocess_interpoleer(loc_528_F6C, rows_with_nan_528_6)
preprocess_interpoleer(loc_528_F18C, rows_with_nan_528_18)
preprocess_interpoleer(loc_528r_F6C, rows_with_nan_528r_6)
preprocess_interpoleer(loc_528r_F18C, rows_with_nan_528r_18)

preprocess_interpoleer(loc_530_F6C, rows_with_nan_530_6)
preprocess_interpoleer(loc_530_F18C, rows_with_nan_530_18)
preprocess_interpoleer(loc_530r_F6C, rows_with_nan_530r_6)
preprocess_interpoleer(loc_530r_F18C, rows_with_nan_530r_18)

preprocess_interpoleer(loc_531_F6C, rows_with_nan_531_6)
preprocess_interpoleer(loc_531_F18C, rows_with_nan_531_18)
preprocess_interpoleer(loc_531r_F6C, rows_with_nan_531r_6)
preprocess_interpoleer(loc_531r_F18C, rows_with_nan_531r_18)

# %% Inconsistency in observations intensity and observations velocity
#   1. Check if observation intensity > 0 and observation velocity ==0
#       --> insert max veloctity of the last and next hour
#       --> max velocity because it is one vehicle so the velocity will not be limited by surrounding vehicles,
#            maybe only by weather conditions, therefore not allowed velocity inserted
#       --> because the max is inserted, this should be done after filtering the outliers.

[loc_501errors_neg_value_F6C,loc_501errors_no_velocity_F6C,loc_501errors_no_intensity_F6C, loc_501median_F6C]=preprocess_missing_vel(loc_501_F6C)
[loc_501errors_neg_value_F18C,loc_501errors_no_velocity_F18C,loc_501errors_no_intensity_F18C, loc_501median_F18C]=preprocess_missing_vel(loc_501_F18C)
[loc_501rerrors_neg_value_F6C,loc_501rerrors_no_velocity_F6C,loc_501rerrors_no_intensity_F6C, loc_501rmedian_F6C]=preprocess_missing_vel(loc_501r_F6C)
[loc_501rerrors_neg_value_F18C,loc_501rerrors_no_velocity_F18C,loc_501rerrors_no_intensity_F18C, loc_501rmedian_F18C]=preprocess_missing_vel(loc_501r_F18C)

[loc_502errors_neg_value_F6C,loc_502errors_no_velocity_F6C,loc_502errors_no_intensity_F6C, loc_502median_F6C]=preprocess_missing_vel(loc_502_F6C)
[loc_502rerrors_neg_value_F6C,loc_502rerrors_no_velocity_F6C,loc_502rerrors_no_intensity_F6C, loc_502rmedian_F6C]=preprocess_missing_vel(loc_502r_F6C)

[loc_528errors_neg_value_F6C,loc_528errors_no_velocity_F6C,loc_528errors_no_intensity_F6C, loc_528median_F6C]=preprocess_missing_vel(loc_528_F6C)
[loc_528errors_neg_value_F18C,loc_528errors_no_velocity_F18C,loc_528errors_no_intensity_F18C, loc_528median_F18C]=preprocess_missing_vel(loc_528_F18C)
[loc_528rerrors_neg_value_F6C,loc_528rerrors_no_velocity_F6C,loc_528rerrors_no_intensity_F6C, loc_528rmedian_F6C]=preprocess_missing_vel(loc_528r_F6C)
[loc_528rerrors_neg_value_F18C,loc_528rerrors_no_velocity_F18C,loc_528rerrors_no_intensity_F18C, loc_528rmedian_F18C]=preprocess_missing_vel(loc_528r_F18C)

[loc_530errors_neg_value_F6C,loc_530errors_no_velocity_F6C,loc_530errors_no_intensity_F6C, loc_530median_F6C]=preprocess_missing_vel(loc_530_F6C)
[loc_530errors_neg_value_F18C,loc_530errors_no_velocity_F18C,loc_530errors_no_intensity_F18C, loc_530median_F18C]=preprocess_missing_vel(loc_530_F18C)
[loc_530rerrors_neg_value_F6C,loc_530rerrors_no_velocity_F6C,loc_530rerrors_no_intensity_F6C, loc_530rmedian_F6C]=preprocess_missing_vel(loc_530r_F6C)
[loc_530rerrors_neg_value_F18C,loc_530rerrors_no_velocity_F18C,loc_530rerrors_no_intensity_F18C, loc_530rmedian_F18C]=preprocess_missing_vel(loc_530r_F6C)

[loc_531errors_neg_value_F6C,loc_531errors_no_velocity_F6C,loc_531errors_no_intensity_F6C, loc_531median_F6C]=preprocess_missing_vel(loc_531_F6C)
[loc_531errors_neg_value_F18C,loc_531errors_no_velocity_F18C,loc_531errors_no_intensity_F18C, loc_531median_F18C]=preprocess_missing_vel(loc_531_F18C)
[loc_531rerrors_neg_value_F6C,loc_531rerrors_no_velocity_F6C,loc_531rerrors_no_intensity_F6C, loc_531rmedian_F6C]=preprocess_missing_vel(loc_531r_F6C)
[loc_531rerrors_neg_value_F18C,loc_531rerrors_no_velocity_F18C,loc_531rerrors_no_intensity_F18C, loc_531rmedian_F18C]=preprocess_missing_vel(loc_531r_F18C)


# %% Print number of days remaining:
print('Number of dates removed in data')
print(     'loc 501 F6C:',365*3- len(loc_501_F6C['start_datum'].unique()))          #30
print(      'loc 501 F18C:', 365*3- len(loc_501_F18C['start_datum'].unique()))      #30
print(      'loc 501r F6C:', 365*3- len(loc_501r_F6C['start_datum'].unique()))      #30
print(      'loc 501r F18C:', 365*3- len(loc_501r_F18C['start_datum'].unique()))    #30

print(     'loc 502 F6C:',365*3-  len(loc_502_F6C['start_datum'].unique()))         #35
print(      'loc 502r F6C:', 365*3- len(loc_502r_F6C['start_datum'].unique()))      #36

print(     'loc 528 F6C:',365*3-  len(loc_528_F6C['start_datum'].unique()))         #32
print(      'loc 528 F18C:',365*3-  len(loc_528_F18C['start_datum'].unique()))      #32
print(      'loc 528r F6C:', 365*3- len(loc_528r_F6C['start_datum'].unique()))      #44
print(      'loc 528r F18C:',365*3-  len(loc_528r_F18C['start_datum'].unique()))    #70    

print(     'loc 530 F6C:', 365*3- len(loc_530_F6C['start_datum'].unique()))         #45
print(      'loc 530 F18C:',365*3-  len(loc_530_F18C['start_datum'].unique()))      #30
print(      'loc 530r F6C:',365*3-  len(loc_530r_F6C['start_datum'].unique()))      #31
print(      'loc 530r F18C:',365*3-  len(loc_530r_F18C['start_datum'].unique()))    #30

print(     'loc 531 F6C:', 365*3- len(loc_531_F6C['start_datum'].unique()))         #30
print(      'loc 531 F18C:',365*3- len(loc_531_F18C['start_datum'].unique()))       #54
print(      'loc 531r F6C:',365*3-  len(loc_531r_F6C['start_datum'].unique()))      #30
print(      'loc 531r F18C:', 365*3- len(loc_531r_F18C['start_datum'].unique()))    #30

''' 
loc_530r_F6C: 306432 rows
    '''
# %%Plot the days that are removed in the data
import calplot
from matplotlib.colors import LinearSegmentedColormap
wit=[1,1,1]

loc_530r_F6C['not_removed']=1
loc_530r_F6C_calplot=loc_530r_F6C.set_index('start_datum')
   
cmap1= LinearSegmentedColormap.from_list("", [siemens_groen,wit])
calplot.calplot(loc_530r_F6C_calplot['not_removed'],cmap=cmap1,textfiller='-',colorbar=False,suptitle='Days removed for location 530r and lane 1') #hier nice kleur map toevoegen maar ligt aan hoeveel clusters



#%% What to do with only zero measurements
# als meer dan 4 uur achter elkaar nul sla de dag op
def check_zero_days(data):
    check=[]
    for i in range(len(data['start_datum'].unique())*4):
        #print((loc_530r_F6C['waarnemingen_intensiteit'][i*288:(i+1)*288]==0).all())
        if (data['waarnemingen_intensiteit'][i*72:(i+1)*72]==0).all():
            check.append(data.loc[data.index[i*72],'start_datum'])
    #print('number of days with 12 hours zero',len(check))
    check_df=pd.DataFrame({'data_zero':check})
    print('number of days with 12 hours zero unique',len(check_df['data_zero'].unique()))
    check_df=check_df.drop_duplicates().reset_index(drop=True) # only keep unique days

    return check_df


check_501_F6C=check_zero_days(loc_501_F6C)              #0
check_501_F18C=check_zero_days(loc_501_F18C)            #0
check_501r_F6C=check_zero_days(loc_501r_F6C)            #0
check_501r_F18C=check_zero_days(loc_501r_F18C)          #0

check_502_F6C=check_zero_days(loc_502_F6C)              #31
check_502r_F6C=check_zero_days(loc_502r_F6C)            #38

check_528_F6C=check_zero_days(loc_528_F6C)              #20
check_528_F18C=check_zero_days(loc_528_F18C)            #0
check_528r_F6C=check_zero_days(loc_528r_F6C)            #22
check_528r_F18C=check_zero_days(loc_528r_F18C)          #28

check_530_F6C=check_zero_days(loc_530_F6C)              #138
check_530_F18C=check_zero_days(loc_530_F18C)            #17
check_530r_F6C=check_zero_days(loc_530r_F6C)            #218
check_530r_F18C=check_zero_days(loc_530r_F18C)          #49

check_531_F6C=check_zero_days(loc_531_F6C)              #2
check_531_F18C=check_zero_days(loc_531_F18C)            #10
check_531r_F6C=check_zero_days(loc_531r_F6C)            #0
check_531r_F18C=check_zero_days(loc_531r_F18C)          #0

# %%Plot the days that are removed in the data
import calplot
from matplotlib.colors import LinearSegmentedColormap
wit=[1,1,1]

def plot_zero_days(data,check):
    data['not_removed']=1
    
    Condition=data['start_datum'].isin(check['data_zero'])
    data.loc[data.index[Condition],'not_removed']=2
    
    data_calplot=data.set_index('start_datum')
       
    cmap1= LinearSegmentedColormap.from_list("", [wit,siemens_groen_light3,siemens_groen])
    calplot.calplot(data_calplot['not_removed'],cmap=cmap1,textfiller='-',colorbar=False,suptitle='Days removed') #hier nice kleur map toevoegen maar ligt aan hoeveel clusters
    #plt.savefig("dataremoved_loc528_2.svg")

all_data=[loc_501_F6C,loc_501_F18C,loc_501r_F6C,loc_501r_F18C,loc_502_F6C,loc_502r_F6C,loc_528_F6C,loc_528_F18C,loc_528r_F6C,loc_528r_F18C,loc_530_F6C,loc_530_F18C,loc_530r_F6C,loc_530r_F18C,loc_531_F6C,loc_531_F18C,loc_531r_F6C,loc_531r_F18C]
all_check=[check_501_F6C,check_501_F18C,check_501r_F6C,check_501r_F18C,check_502_F6C,check_502r_F6C,check_528_F6C,check_528_F18C,check_528r_F6C,check_528r_F18C,check_530_F6C,check_530_F18C,check_530r_F6C,check_530r_F18C,check_531_F6C,check_531_F18C,check_531r_F6C,check_531r_F18C]

for i in range(18):
    plot_zero_days(all_data[i], all_check[i])


# days removed door filtering: not in data fillcolor --> wit
# days still in data : not removed = 1               --> licht groen
# days that are zero : not removed = 2,              --> siemens_groen

#%% Remove zero days. 

def remove_zero_days(data,check):
    data_new=data.copy(deep=True)
    print('data before',len(data_new['start_datum'].unique()))
    for i in range(len(check)):
        indexes_day=data_new[data_new['start_datum']==check.loc[i,'data_zero']].index #corresponding indexes
        data_new=data_new.drop(index=indexes_day,inplace=False) #inplace false return a copy
    
    print('data after',len(data_new['start_datum'].unique()))
        
    return data_new

loc_501_F6C_zero=remove_zero_days(loc_501_F6C,check_501_F6C)
loc_501_F18C_zero=remove_zero_days(loc_501_F18C,check_501_F18C)
loc_501r_F6C_zero=remove_zero_days(loc_501r_F6C,check_501r_F6C)
loc_501r_F18C_zero=remove_zero_days(loc_501r_F18C,check_501r_F18C)

loc_502_F6C_zero=remove_zero_days(loc_502_F6C,check_502_F6C)
loc_502r_F6C_zero=remove_zero_days(loc_502r_F6C,check_502r_F6C)

loc_528_F6C_zero=remove_zero_days(loc_528_F6C,check_528_F6C)
loc_528_F18C_zero=remove_zero_days(loc_528_F18C,check_528_F18C)
loc_528r_F6C_zero=remove_zero_days(loc_528r_F6C,check_528r_F6C)
loc_528r_F18C_zero=remove_zero_days(loc_528r_F18C,check_528r_F18C)

loc_530_F6C_zero=remove_zero_days(loc_530_F6C,check_530_F6C)
loc_530_F18C_zero=remove_zero_days(loc_530_F18C,check_530_F18C)
loc_530r_F6C_zero=remove_zero_days(loc_530r_F6C,check_530r_F6C)
loc_530r_F18C_zero=remove_zero_days(loc_530r_F18C,check_530r_F18C)

loc_531_F6C_zero=remove_zero_days(loc_531_F6C,check_531_F6C)
loc_531_F18C_zero=remove_zero_days(loc_531_F18C,check_531_F18C)
loc_531r_F6C_zero=remove_zero_days(loc_531r_F6C,check_531r_F6C)
loc_531r_F18C_zero=remove_zero_days(loc_531r_F18C,check_531r_F18C)


#%% Make new dataframe to plot
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import calplot

wit=[1,1,1]

def plot_calplot_removed_2(data,check,loc,lane):
    if check.size==0:
        rng = pd.date_range('2017-01-01', periods=365*3, freq='D')
        data_to_plot = pd.DataFrame({ 'start_datum': rng}) 
        
        # fill in  everything with zeros
        data_to_plot['not_removed']=0
        
        # fill in with one if removed by filtering
        Condition1=~data_to_plot['start_datum'].isin(data['start_datum'].unique())
        
        data_to_plot.loc[data_to_plot.index[Condition1],'not_removed']=1
        
        data_calplot=data_to_plot.set_index('start_datum')
        cmap1= LinearSegmentedColormap.from_list("", [siemens_groen_light2,siemens_groen_light3])
        
        calplot.calplot(data_calplot['not_removed'],cmap=cmap1,textfiller='-',dropzero=False,colorbar=False,suptitle='Days removed location {} and lane {} '.format(loc,lane)) #hier nice kleur map toevoegen maar ligt aan hoeveel clusters
        plt.savefig("dataremoved_{}_{}.svg".format(loc,lane))     
    else:
        rng = pd.date_range('2017-01-01', periods=365*3, freq='D')
        data_to_plot = pd.DataFrame({ 'start_datum': rng}) 
        
        # fill in  everything with zeros
        data_to_plot['not_removed']=0
        
        # fill in with one if removed by filtering
        Condition1=~data_to_plot['start_datum'].isin(data['start_datum'].unique())
        
        data_to_plot.loc[data_to_plot.index[Condition1],'not_removed']=1
        
        # fill in with number 2 if removed because zero
        Condition2=data_to_plot['start_datum'].isin(check['data_zero'])
        data_to_plot.loc[data_to_plot.index[Condition2],'not_removed']=2
        
        data_calplot=data_to_plot.set_index('start_datum')
               
        cmap1= LinearSegmentedColormap.from_list("", [siemens_groen_light2,siemens_groen])
        
        calplot.calplot(data_calplot['not_removed'],cmap=cmap1,textfiller='-',dropzero=False,colorbar=False,suptitle='Days removed location {} and lane {} '.format(loc,lane)) #hier nice kleur map toevoegen maar ligt aan hoeveel clusters
        plt.savefig("dataremoved_{}_{}.svg".format(loc,lane))     


plot_calplot_removed_2(loc_501_F6C,check_501_F6C, '501','1')
plot_calplot_removed_2(loc_501_F18C,check_501_F18C,'501','2')
plot_calplot_removed_2(loc_501r_F6C,check_501r_F6C,'501r','1')
plot_calplot_removed_2(loc_501r_F18C,check_501r_F18C,'501r','2')

plot_calplot_removed_2(loc_502_F6C,check_502_F6C,'502','1')
plot_calplot_removed_2(loc_502r_F6C,check_502r_F6C,'502r','2')

plot_calplot_removed_2(loc_528_F6C,check_528_F6C,'528','1')
plot_calplot_removed_2(loc_528_F18C,check_528_F18C,'528','2')
plot_calplot_removed_2(loc_528r_F6C,check_528r_F6C,'528r','1')
plot_calplot_removed_2(loc_528r_F18C,check_528r_F18C,'528r','2')

plot_calplot_removed_2(loc_530_F6C,check_530_F6C,'530','1')
plot_calplot_removed_2(loc_530_F18C,check_530_F18C,'530','2')
plot_calplot_removed_2(loc_530r_F6C,check_530r_F6C,'530r','1')
plot_calplot_removed_2(loc_530r_F18C,check_530r_F18C,'530r','2')

plot_calplot_removed_2(loc_531_F6C,check_531_F6C,'531','1')
plot_calplot_removed_2(loc_531_F18C,check_531_F18C,'531','2')
plot_calplot_removed_2(loc_531r_F6C,check_531r_F6C,'531r','1')
plot_calplot_removed_2(loc_531r_F18C,check_531r_F18C,'531r','2')

# %% print number of days removed in total
print('Number of dates removed in data')
print(     'loc 501 F6C:',(365*3- len(loc_501_F6C_zero['start_datum'].unique())))
print(      'loc 501 F18C:',( 365*3- len(loc_501_F18C_zero['start_datum'].unique())))
print(      'loc 501r F6C:',( 365*3- len(loc_501r_F6C_zero['start_datum'].unique())))
print(      'loc 501r F18C:', (365*3- len(loc_501r_F18C_zero['start_datum'].unique())))

print(     'loc 502 F6C:',(365*3-  len(loc_502_F6C_zero['start_datum'].unique())))
print(      'loc 502r F6C:', (365*3- len(loc_502r_F6C_zero['start_datum'].unique())))

print(     'loc 528 F6C:',(365*3-  len(loc_528_F6C_zero['start_datum'].unique())))
print(      'loc 528 F18C:',(365*3-  len(loc_528_F18C_zero['start_datum'].unique())))
print(      'loc 528r F6C:', (365*3- len(loc_528r_F6C_zero['start_datum'].unique())))
print(      'loc 528r F18C:',(365*3-  len(loc_528r_F18C_zero['start_datum'].unique())))

print(     'loc 530 F6C:', (365*3- len(loc_530_F6C_zero['start_datum'].unique())))
print(      'loc 530 F18C:',(365*3-  len(loc_530_F18C_zero['start_datum'].unique())))
print(      'loc 530r F6C:',(365*3-  len(loc_530r_F6C_zero['start_datum'].unique())))
print(      'loc 530r F18C:',(365*3-  len(loc_530r_F18C_zero['start_datum'].unique())))

print(     'loc 531 F6C:',( 365*3- len(loc_531_F6C_zero['start_datum'].unique())))
print(      'loc 531 F18C:',(365*3- len(loc_531_F18C_zero['start_datum'].unique())))
print(      'loc 531r F6C:',(365*3-  len(loc_531r_F6C_zero['start_datum'].unique())))
print(      'loc 531r F18C:', (365*3- len(loc_531r_F18C_zero['start_datum'].unique())))

# %% Save all data
#COMMENTED OUT SUCH THAT DOES NOT HAPPEN ON ACCIDENT
'''
loc_501_F6C_zero.to_csv('data_preprocessed\loc_501_F6C.csv')
loc_501_F18C_zero.to_csv('data_preprocessed\loc_501_F18C.csv')
loc_501r_F6C_zero.to_csv('data_preprocessed\loc_501r_F6C.csv')
loc_501r_F18C_zero.to_csv('data_preprocessed\loc_501r_F18C.csv')
loc_502_F6C_zero.to_csv('data_preprocessed\loc_502_F6C.csv')
loc_502r_F6C_zero.to_csv('data_preprocessed\loc_502r_F6C.csv')
loc_528_F6C_zero.to_csv('data_preprocessed\loc_528_F6C.csv')
loc_528_F18C_zero.to_csv('data_preprocessed\loc_528_F18C.csv')
loc_528r_F6C_zero.to_csv('data_preprocessed\loc_528r_F6C.csv')
loc_528r_F18C_zero.to_csv('data_preprocessed\loc_528r_F18C.csv')
loc_530_F6C_zero.to_csv('data_preprocessed\loc_530_F6C.csv')
loc_530_F18C_zero.to_csv('data_preprocessed\loc_530_F18C.csv')
loc_530r_F6C_zero.to_csv('data_preprocessed\loc_530r_F6C.csv')
loc_530r_F18C_zero.to_csv('data_preprocessed\loc_530r_F18C.csv')
loc_531_F6C_zero.to_csv('data_preprocessed\loc_531_F6C.csv')
loc_531_F18C_zero.to_csv('data_preprocessed\loc_531_F18C.csv')
loc_531r_F6C_zero.to_csv('data_preprocessed\loc_531r_F6C.csv')
loc_531r_F18C_zero.to_csv('data_preprocessed\loc_531r_F18C.csv')

'''









