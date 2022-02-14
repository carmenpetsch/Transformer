# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:28:42 2021

@author: z0049unj

Investigate weather correlations with traffic flow
"""
# %% Import packages
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix

# to plot figures
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# clustering
import calplot
from sklearn.cluster import AgglomerativeClustering
from scipy import stats # for z-score normalization

# import functions from baseline_models_functions
from baseline_models_functions import plot_clusters
from baseline_models_functions import median_dow

# machine learning models
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Upload Feature space
X_501_train_time=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_time.pkl")
X_501_train_dow=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_dow.pkl")
X_501_train_season=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_season.pkl")
X_501_train_feestdag=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_feestdag.pkl")
X_501_test_feestdag=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_feestdag_test.pkl")

#upload traffic flow train
y_501_train=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/y_501_train.pkl")

#upload weather data
Wind_data_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/Weer/Wind_data_2019.csv')#,index_col=0)
Temp_data_2019=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/Weer/Temp_data_2019.csv')#,index_col=0)

Wind_data_all=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/Weer/Wind_data_all.csv')#,index_col=0)
Temp_data_all=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/Data/Weer/Temp_data_all.csv')#,index_col=0)

#upload all traffic flow all locations
loc_501=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_501.pkl")
loc_501r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_501r.pkl")
loc_502=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_502.pkl")
loc_502r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_502r.pkl")
loc_528=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_528.pkl")
loc_528r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_528r.pkl")
loc_530=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_530.pkl")
loc_530r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_530r.pkl")
loc_531=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_531.pkl")
loc_531r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_531r.pkl")

# opload index before preprocessing Required to remove the corresponding weahter data aswell
original_index_501=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_501.pkl")
original_index_501r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_501r.pkl")

original_index_502=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_502.pkl")
original_index_502r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_502r.pkl")

original_index_528=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_528.pkl")
original_index_528r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_528r.pkl")

original_index_530=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_530.pkl")
original_index_530r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_530r.pkl")


original_index_531=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_531.pkl")
original_index_531r=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_531r.pkl")

# make color map
b3=(64/255, 97/255, 188/255)
o1=(242/255, 98/255, 0/255)

o3=(253/255, 131/255, 65/255)
wit=(1,1,1)

siemens_groen=(0/255,153/255,153/255)
siemens_groen_light1=(92/255,176/255,175/255)
siemens_groen_light2=(142/255,198/255,197/255)
siemens_groen_light3=(188/255,221/255,220/255)
siemens_blauw=(1/255,8/255,46/255)
cmap_2_colors=LinearSegmentedColormap.from_list("", [o1,wit,siemens_groen])


#%% Only extract relevant columns
Wind_relevant=Wind_data_all[['DD','FH','FF','FX']]
Temp_relevant=Temp_data_all[['T','TD','SQ','Q','DR','RH','U']]
Weather_relevant=pd.concat([Wind_relevant,Temp_relevant],axis=1)


Weather_relevant=Weather_relevant.rename(columns={'DD':'Wind direction','FH':'Average hourly windspeed','FF':'Average windspeed last 10 min','FX':'Wind peak','T':'Temperature','TD':'Dew temperature','SQ':'Sun duration','Q':'Radiation','DR':'Precipitation duration','RH':'Precipitation sum','U':'Relative humidity'})
Weather_relevant['Temperature']=Weather_relevant['Temperature']/10              #such that converted to degrees
Weather_relevant['Dew temperature']=Weather_relevant['Dew temperature']/10
# Fix summer and winter time
# at 31 march 02:00:00 is het eigenlijk 03:00:00
# voeg op 02:00:00 dezelfde rij nog een keer toe

index_2017=loc_501.index[loc_501['start_datum']=='2017-3-26'][2]
index_2018=loc_501.index[loc_501['start_datum']=='2018-3-25'][2]
index_2019=loc_501.index[loc_501['start_datum']=='2019-3-31'][2]
insert_df_2017=pd.DataFrame(Weather_relevant.loc[index_2017,:]).T #89*24+2
insert_df_2018=pd.DataFrame(Weather_relevant.loc[index_2018,:]).T 
insert_df_2019=pd.DataFrame(Weather_relevant.loc[index_2019,:]).T 

Weather_relevant=pd.concat([Weather_relevant.iloc[:index_2017], insert_df_2017, Weather_relevant.iloc[index_2017:index_2018],insert_df_2018,Weather_relevant.iloc[index_2018:index_2019],insert_df_2019,Weather_relevant.iloc[index_2019:]]).reset_index(drop=True)

# op 27/10 om 03:00:00 wordt het weer 02:00:00 en klopt het weer
# verwijder hier de rij van 02:00:00
drop_2017=loc_501.index[loc_501['start_datum']=='2017-10-29'][2]
drop_2018=loc_501.index[loc_501['start_datum']=='2018-10-28'][2]
drop_2019=loc_501.index[loc_501['start_datum']=='2019-10-27'][2]
Weather_relevant=Weather_relevant.drop([drop_2017]).reset_index()
del Weather_relevant['index']
Weather_relevant=Weather_relevant.drop([drop_2018]).reset_index()
del Weather_relevant['index']
Weather_relevant=Weather_relevant.drop([drop_2019]).reset_index()
del Weather_relevant['index']

# %%remove days that are removed in traffic data in preprocessing step
Weather_preprocessed_501=Weather_relevant.loc[original_index_501['original_index'],:].reset_index()
Weather_preprocessed_501r=Weather_relevant.loc[original_index_501r['original_index'],:].reset_index()

Weather_preprocessed_502=Weather_relevant.loc[original_index_502['original_index'],:].reset_index()
Weather_preprocessed_502r=Weather_relevant.loc[original_index_502r['original_index'],:].reset_index()

Weather_preprocessed_528=Weather_relevant.loc[original_index_528['original_index'],:].reset_index()
Weather_preprocessed_528r=Weather_relevant.loc[original_index_528r['original_index'],:].reset_index()

Weather_preprocessed_530=Weather_relevant.loc[original_index_530['original_index'],:].reset_index()
Weather_preprocessed_530r=Weather_relevant.loc[original_index_530r['original_index'],:].reset_index()

Weather_preprocessed_531=Weather_relevant.loc[original_index_531['original_index'],:].reset_index()
Weather_preprocessed_531r=Weather_relevant.loc[original_index_531r['original_index'],:].reset_index()


del Weather_preprocessed_501['index']
del Weather_preprocessed_501r['index']
del Weather_preprocessed_502['index']
del Weather_preprocessed_502r['index']
del Weather_preprocessed_528['index']
del Weather_preprocessed_528r['index']
del Weather_preprocessed_530['index']
del Weather_preprocessed_530r['index']
del Weather_preprocessed_531['index']
del Weather_preprocessed_531r['index']


# %% Look into location 501
def plot_corr_matrix(corr_matrix,corrtype,loc):
    # plot correlation map
    plt.figure()
    plt.title(' Spearman\'s rank coefficient location {} '.format(loc))
    ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,cmap=cmap_2_colors,square=corrtype)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right');
    
    plt.savefig('{}_{}.svg'.format(corrtype,loc))  

def corr_matrix_weather(data,weather_data,corrtype,loc):
    Data_total=pd.concat([data['waarnemingen_intensiteit'],weather_data],axis=1)
    Data_total=Data_total.rename(columns={'waarnemingen_intensiteit':'Traffic flow'})
    
    # linear or non linear correlation, depends on type
    corr_matrix=Data_total.corr(method=corrtype)

    plot_corr_matrix(corr_matrix,corrtype,loc)
    return Data_total

Data_total_501=corr_matrix_weather(loc_501,Weather_preprocessed_501,'spearman','501')

#%%
Data_total_501r=corr_matrix_weather(loc_501r,Weather_preprocessed_501r,'spearman','501r')

Data_total_502=corr_matrix_weather(loc_502,Weather_preprocessed_502,'spearman','502')
Data_total_502r=corr_matrix_weather(loc_502r,Weather_preprocessed_502r,'spearman','502r')

Data_total_528=corr_matrix_weather(loc_528,Weather_preprocessed_528,'spearman','528')
Data_total_528r=corr_matrix_weather(loc_528r,Weather_preprocessed_528r,'spearman','528r')

Data_total_530=corr_matrix_weather(loc_530,Weather_preprocessed_530,'spearman','530')
Data_total_530r=corr_matrix_weather(loc_530r,Weather_preprocessed_530r,'spearman','530r')

Data_total_531=corr_matrix_weather(loc_531,Weather_preprocessed_531,'spearman','531')
Data_total_531r=corr_matrix_weather(loc_531r,Weather_preprocessed_531r,'spearman','531r')


#print(corr_matrix_spearman.to_latex(index=False)) # pring to latex format
#scatter_matrix(Data_501[['Traffic flow','Temperature','Dew temperature','Sun duration','Radiation','Precipitation duration','Relative humidity']],alpha=0.3,color=b3,diagonal='hist',hist_kwds={'bins': 60})

#%% plot scatter figure
# Traffic flow and temperature
plt.figure()
plt.scatter(Data_total_501['Temperature'],Data_total_501['Traffic flow'],color=siemens_groen,alpha=0.05)
plt.xlabel('Temperature [$^\circ$C]')
plt.ylabel('Traffic flow [veh/h]')
plt.title('Correlation between traffic flow and temperature')
plt.savefig('Relation_flow_temp.svg')  

plt.figure()
plt.scatter(Data_total_501['Radiation'],Data_total_501['Traffic flow'],color=siemens_groen,alpha=0.05)
plt.xlabel('Radiation [J/cm$^2$]')
plt.ylabel('Traffic flow [veh/h]')
plt.title('Correlation between traffic flow and radiation')
plt.savefig('Relation_flow_rad.svg')  

plt.figure()
plt.scatter(Data_total_501['Relative humidity'],Data_total_501['Traffic flow'],color=siemens_groen,alpha=0.05)
plt.xlabel('Relative humidity [%]')
plt.ylabel('Traffic flow [veh/h]')
plt.title('Correlation between traffic flow and relative humidity')
plt.savefig('Relation_flow_rel_hum.svg')  

plt.figure()
plt.scatter(Data_total_501['Dew temperature'],Data_total_501['Traffic flow'],color=siemens_groen,alpha=0.05)
plt.xlabel('Radiation [J/cm$^2$]')
plt.ylabel('Traffic flow [veh/h]')
plt.title('Correlation between traffic flow and radiation')

# %% Investigate the type of correlation

def plot_scatter_matrix(data,loc):

    axes=scatter_matrix(data[['Traffic flow','Temperature','Sun duration','Radiation','Relative humidity']],alpha=0.05,diagonal='hist',hist_kwds={'bins': 40,'color':siemens_groen},color=siemens_groen_light1)
    #plt.title('Correlation analysis location {}'.format(loc))
   
    for ax in axes.flatten():
        ax.xaxis.label.set_rotation(30)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    
    #plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()
    plt.savefig('scatter_plot_weather_loc{}.svg'.format(loc))

plot_scatter_matrix(Data_total_501,'501')
plot_scatter_matrix(Data_total_501r,'501r')

plot_scatter_matrix(Data_total_502,'502')
plot_scatter_matrix(Data_total_502r,'502r')

plot_scatter_matrix(Data_total_528,'528')
plot_scatter_matrix(Data_total_528r,'528r')

plot_scatter_matrix(Data_total_530,'530')
plot_scatter_matrix(Data_total_530r,'530r')

plot_scatter_matrix(Data_total_531,'531')
plot_scatter_matrix(Data_total_531r,'531r')


#%% Hebben week en weekend dagen een zelfde soort correlatie met weather features?
def correlation_week_weekend(data_total,data,corrtype,loc):
    Data_weekday=pd.concat([data_total,data['weekday']],axis=1)
    
    Data_week=Data_weekday.loc[Data_weekday.index[Data_weekday['weekday']<=4],['Traffic flow', 'Wind direction', 'Average hourly windspeed',
           'Average windspeed last 10 min', 'Wind peak', 'Temperature',
           'Dew temperature', 'Sun duration', 'Radiation',
           'Precipitation duration', 'Precipitation sum', 'Relative humidity']].reset_index()
    Data_weekend=Data_weekday.loc[Data_weekday.index[Data_weekday['weekday']>4],['Traffic flow', 'Wind direction', 'Average hourly windspeed',
           'Average windspeed last 10 min', 'Wind peak', 'Temperature',
           'Dew temperature', 'Sun duration', 'Radiation',
           'Precipitation duration', 'Precipitation sum', 'Relative humidity']].reset_index()
    
    del Data_week['index']
    del Data_weekend['index']
    
    # Perform the same correlations analysis
    corr_matrix_week=Data_week.corr(method=corrtype)
    corr_matrix_weekend=Data_weekend.corr(method=corrtype)
    
    plot_corr_matrix(corr_matrix_week,'{} week'.format(corrtype),loc)
    plot_corr_matrix(corr_matrix_weekend,'{} weekend'.format(corrtype),loc)
    
    plot_scatter_matrix(Data_week,'{} week'.format(loc))
    plot_scatter_matrix(Data_weekend,'{} weekend'.format(loc))

correlation_week_weekend(Data_total_501,loc_501,'pearson','501')
correlation_week_weekend(Data_total_501r,loc_501r,'pearson','501r')

correlation_week_weekend(Data_total_502,loc_502,'pearson','502')
correlation_week_weekend(Data_total_502r,loc_502r,'pearson','502r')

correlation_week_weekend(Data_total_528,loc_528,'pearson','528')
correlation_week_weekend(Data_total_528r,loc_528r,'pearson','528r')

correlation_week_weekend(Data_total_530,loc_530,'pearson','530')
correlation_week_weekend(Data_total_530r,loc_530r,'pearson','530r')

correlation_week_weekend(Data_total_531,loc_531,'pearson','531')
correlation_week_weekend(Data_total_531r,loc_531r,'pearson','531r')

# %% Probeer het alleen nog per dag, misschien een verschil
def correlation_all_dow(data_total,data,corrtype,loc):
    corr_array=pd.DataFrame()
    Data_with_weekday=pd.concat([data_total,data['weekday']],axis=1)

    for i in range(7):     
        Data_subtracted=Data_with_weekday.loc[Data_with_weekday.index[Data_with_weekday['weekday']==i],['Traffic flow', 'Wind direction', 'Average hourly windspeed',
               'Average windspeed last 10 min', 'Wind peak', 'Temperature',
               'Dew temperature', 'Sun duration', 'Radiation',
               'Precipitation duration', 'Precipitation sum', 'Relative humidity']].reset_index()
    
        del Data_subtracted['index']
        
        # (non) linear correlations
        corr_matrix_day=Data_subtracted.corr(method=corrtype)
        
       # plot_corr_matrix(corr_matrix_day,'{} day {}'.format(corrtype,i),loc)
        #plot_scatter_matrix(Data_subtracted,'{} day {}'.format(loc,i))

        corr_new=corr_matrix_day.loc['Traffic flow',:]
    
        corr_array=pd.concat([corr_array,corr_new],axis=1)

    corr_array_day=corr_array.drop('Traffic flow')
    
    corr_array_day.columns=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    
    plt.figure()
    corr_array_day.plot.bar()
    plt.xticks(rotation=90)
    plt.title('{} correlation per day of the week location {}'.format(corrtype,loc))
    plt.ylabel('Correlation coefficient')
    plt.yticks(np.arange(-1, 1.2,0.2))


correlation_all_dow(Data_total_501,loc_501,'pearson','501')
correlation_all_dow(Data_total_501r,loc_501r,'pearson','501r')
#%%
correlation_all_dow(Data_total_502,loc_502,'pearson','502')
correlation_all_dow(Data_total_502r,loc_502r,'pearson','502r')

correlation_all_dow(Data_total_528,loc_528,'pearson','528')
correlation_all_dow(Data_total_528r,loc_528r,'pearson','528r')

correlation_all_dow(Data_total_530,loc_530,'pearson','530')
correlation_all_dow(Data_total_530r,loc_530r,'pearson','530r')

correlation_all_dow(Data_total_531,loc_531,'pearson','531')
correlation_all_dow(Data_total_531r,loc_531r,'pearson','531r')


# %% Look into temperature
plt.figure()
plt.scatter(Data_total_501['Temperature'],Data_total_501['Traffic flow'],alpha=0.1)
plt.xlabel('Temperature')
plt.ylabel('Traffic flow')

# %%subtract median of the day --> niet echt verschil ik denk ook niet echt nuttig, weather hangt ook af van tijd 
traffic_flow_data_501=pd.concat([loc_501['start_tijd'],Data_total_501['Traffic flow']],axis=1)

data_median_501=traffic_flow_data_501.groupby(["start_tijd"]).median().reset_index(drop="False")


loc_501_median_full=np.transpose(np.tile(data_median_501,(1065,))).ravel()  #1065 number days total in 3 years

deseason=pd.Series(traffic_flow_data_501['Traffic flow'].array-loc_501_median_full)

plt.figure()
plt.scatter(Data_total_501['Temperature'],deseason,alpha=0.1)
plt.xlabel('Temperature')
plt.ylabel('Traffic flow median of time subtracted')

# %% subtract daily median and look into correlations, more clear?
def median_dow(data):
    loc_dow=data.groupby(["weekday"])
    data_0=loc_dow.get_group(0)
    data_0=data_0.reset_index(drop="True")
    data_1=loc_dow.get_group(1)
    data_1=data_1.reset_index(drop="True")
    data_2=loc_dow.get_group(2)
    data_2=data_2.reset_index(drop="True")
    data_3=loc_dow.get_group(3)
    data_3=data_3.reset_index(drop="True")
    data_4=loc_dow.get_group(4)
    data_4=data_4.reset_index(drop="True")
    data_5=loc_dow.get_group(5)
    data_5=data_5.reset_index(drop="True")
    data_6=loc_dow.get_group(6)
    data_6=data_6.reset_index(drop="True")    
    
    
    data_0_av=data_0.groupby(["start_tijd"]).median().reset_index(drop="False")
    data_1_av=data_1.groupby(["start_tijd"]).median().reset_index(drop="False")
    data_2_av=data_2.groupby(["start_tijd"]).median().reset_index(drop="False")
    data_3_av=data_3.groupby(["start_tijd"]).median().reset_index(drop="False")
    data_4_av=data_4.groupby(["start_tijd"]).median().reset_index(drop="False")
    data_5_av=data_5.groupby(["start_tijd"]).median().reset_index(drop="False")
    data_6_av=data_6.groupby(["start_tijd"]).median().reset_index(drop="False")    

    return data_0_av['waarnemingen_intensiteit'],data_1_av['waarnemingen_intensiteit'],data_2_av['waarnemingen_intensiteit'],data_3_av['waarnemingen_intensiteit'],data_4_av['waarnemingen_intensiteit'],data_5_av['waarnemingen_intensiteit'],data_6_av['waarnemingen_intensiteit']
   
data_0_av,data_1_av,data_2_av,data_3_av,data_4_av,data_5_av,data_6_av=median_dow(loc_501)

week_median=pd.concat([data_0_av,data_1_av,data_2_av,data_3_av,data_4_av,data_5_av,data_6_av])


week_normal=pd.concat([data_0_av,data_1_av,data_2_av,data_3_av,data_4_av,data_5_av,data_6_av])

week_1=data_6_av
week_between=pd.Series(np.transpose(np.tile(week_normal,(156)))) 
week_end=pd.concat([data_0_av,data_1_av])

week_totaal=pd.concat([week_1,week_between,week_end]).reset_index()
del week_totaal['index']

# median preprocessed
week_totaal_preprocessed_501=week_totaal.loc[original_index_501['original_index'],:]#.reset_index()

# Dow median subtracted 
deseason=pd.Series(loc_501['waarnemingen_intensiteit'].array-week_totaal_preprocessed_501[0])

Data_501_preprocessed=Data_total_501.copy(deep=True)
Data_501_preprocessed['Traffic flow']=deseason

# find correlations
# linear correlation
corr_matrix_pearson=Data_501_preprocessed.corr(method='pearson')
# non linear correlations
corr_matrix_spearman=Data_501_preprocessed.corr(method='spearman')

# plot correlations
plot_corr_matrix(corr_matrix_pearson,'pearson','501')
plt.savefig('figures/corr_matrix_pearson_median.svg')  

plot_corr_matrix(corr_matrix_spearman,'spearman','501')
plt.savefig('figures/corr_matrix_spearman_median.svg')  

#print(corr_matrix_spearman.to_latex(index=False))

# plot scatter matrix
plot_scatter_matrix(Data_501_preprocessed,'501')
plt.savefig('figures/scatter_plot_median.svg')  


# %% Make new feature space with important parameters
Data_total=[Data_total_501,Data_total_501r,Data_total_502,Data_total_502r,Data_total_528,Data_total_528r,Data_total_530,Data_total_530r,Data_total_531,Data_total_531r]
locations=['501','501r','502','502r','528','528r','530','530r','531','531r']



for i in range(len(Data_total)):
    Data_location=Data_total[i]
    X_train_feestdag=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_{}_feestdag.pkl".format(locations[i]))
    X_test_feestdag=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_{}_feestdag_test.pkl".format(locations[i]))
    loc=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_{}.pkl".format(locations[i]))
    print(X_train_feestdag.shape)
    
    data_weather=[Data_location['Temperature'],Data_location['Sun duration'],Data_location['Radiation'],Data_location['Relative humidity'],Data_location['Precipitation duration']]
    headers=['Temperature','Sun_duration','Radiation','Rel_humidity','Precipitation_duration']
    Data_weather=pd.concat(data_weather,axis=1,keys=headers)
        
    X_feestdag_totaal=pd.concat([X_train_feestdag,X_test_feestdag])

    X_weather=pd.concat([X_feestdag_totaal,Data_weather],axis=1)
    
    condition_train=loc['start_datum']<'2019-06-01'
    condition_test=loc['start_datum']>='2019-06-01'
    
    X_weather_train=X_weather.loc[X_weather.index[condition_train],:]
    X_weather_test=X_weather.loc[X_weather.index[condition_test],:]   #negative condition
    '''
    X_weather_train.to_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_{}_weather.pkl".format(locations[i]))
    X_weather_test.to_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_{}_weather_test.pkl".format(locations[i]))
    '''
    
    
    '''
    Data_location=Data_total[i]
    data_weather=[Data_total_501['Temperature'],Data_total_501['Sun duration'],Data_total_501['Radiation'],Data_total_501['Relative humidity'],Data_total_501['Precipitation duration']]
    headers=['Temperature','Sun_duration','Radiation','Rel_humidity','Precipitation_duration']
    Data_501_weather=pd.concat(data_weather,axis=1,keys=headers)
    
    X_feestdag_totaal=pd.concat([X_501_train_feestdag,X_501_test_feestdag])
    X_501_weather=pd.concat([X_feestdag_totaal,Data_501_weather],axis=1)
    
    condition_train=loc_501['start_datum']<'2019-06-01'
    condition_test=loc_501['start_datum']>='2019-06-01'
    
    X_501_weather_train=X_501_weather.loc[X_501_weather.index[condition_train],:]
    X_501_weather_test=X_501_weather.loc[X_501_weather.index[condition_test],:]   #negative condition
    
    
    X_501_weather_train.to_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_weather.pkl")
    X_501_weather_test.to_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Dataframes_features/X_501_weather_test.pkl")
    '''
# %% Can i cluster weather features
# %%by default it is run 10 times the clustering algorithm, als n_init=30 precies hetzelfde
from sklearn.cluster import KMeans

inertia=np.array([])
number_clusters=np.array([])
for i in range(2,20):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(Data_501_weather)
    labels=kmeans.labels_
    print('i',kmeans.inertia_)
    inertia=np.append(inertia,kmeans.inertia_)
    number_clusters=np.append(number_clusters,i)
    # add labels to dataframe
    Data_501_weather['label']=labels
    

# use the elbow method if you do not know the number of clusters
plt.figure()
plt.plot(number_clusters,inertia)
plt.xlabel('number of clusters')
plt.ylabel('inertia')

# oke opzich wel 6 clusters of 9 
#%%
# plot cluster centers of n=6
kmeans6 = KMeans(n_clusters=6, random_state=0).fit(Data_501_weather)
labels=kmeans6.labels_

centers=kmeans6.cluster_centers_
plt.figure()
plt.plot(centers[0],label='cluster0')
plt.plot(centers[1],label='cluster1')
plt.plot(centers[2],label='cluster2')
plt.plot(centers[3],label='cluster3')
plt.plot(centers[4],label='cluster4')
plt.plot(centers[5],label='cluster5')
plt.legend()

# 'Temperature', 'Sun_duration', 'Radiation', 'Rel_humidity','Precipitation_duration'

# %%Ik denk misschien duidelijker in een barplot
x = np.arange(len(centers[0]))  
width=0.1
fig, ax = plt.subplots()
rects1 = ax.bar(x , centers[0], width, label='cluster 0')
rects2 = ax.bar(x + 0.1, centers[1], width, label='cluster 1')
rects3 = ax.bar(x +0.2, centers[2], width, label='cluster 2')
rects4 = ax.bar(x + 0.3, centers[3], width, label='cluster 3')
rects5 = ax.bar(x +0.4, centers[4], width, label='cluster 4')
rects6 = ax.bar(x + 0.5, centers[5], width, label='cluster 5')
labels=['Temperature', 'Sun_duration', 'Radiation', 'Rel_humidity','Precipitation_duration']
ax.set_xticks(x)
ax.set_xticklabels(labels) 
plt.xticks(rotation=80)

#%%

# plot cluster centers of n=9
kmeans9 = KMeans(n_clusters=9, random_state=0).fit(Data_501_weather)
labels=kmeans9.labels_

centers=kmeans9.cluster_centers_
plt.figure()
plt.plot(centers[0],label='cluster0')
plt.plot(centers[1],label='cluster1')
plt.plot(centers[2],label='cluster2')
plt.plot(centers[3],label='cluster3')
plt.plot(centers[4],label='cluster4')
plt.plot(centers[5],label='cluster5')
plt.plot(centers[6],label='cluster6')
plt.legend()

# %%Ik denk misschien duidelijker in een barplot
x = np.arange(len(centers[0]))  
width=0.1
fig, ax = plt.subplots()
rects1 = ax.bar(x , centers[0], width, label='cluster 0')
rects2 = ax.bar(x + 0.1, centers[1], width, label='cluster 1')
rects3 = ax.bar(x +0.2, centers[2], width, label='cluster 2')
rects4 = ax.bar(x + 0.3, centers[3], width, label='cluster 3')
rects5 = ax.bar(x +0.4, centers[4], width, label='cluster 4')
rects6 = ax.bar(x + 0.5, centers[5], width, label='cluster 5')
rects7 = ax.bar(x +0.6, centers[6], width, label='cluster 6')
rects8 = ax.bar(x + 0.7, centers[7], width, label='cluster 7')
rects9 = ax.bar(x +0.8, centers[8], width, label='cluster 8')

labels=['Temperature', 'Sun_duration', 'Radiation', 'Rel_humidity','Precipitation_duration']
ax.set_xticks(x)
ax.set_xticklabels(labels) 
plt.xticks(rotation=80)



