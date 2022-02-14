# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:30:05 2021

@author: z0049unj
"""

''' Figure out untill which prediction horizon the recent data important'''

# %% Import packages
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats # for z-score normalization


# import functions from baseline models function file
# %%Define colors
color_tu=(0, 166/255, 214/255)
#plt.rcParams["font.family"] = "sans-serif"
wit=(1,1,1)
#cmap_blue_orange_2 = LinearSegmentedColormap.from_list("", colors_2)

siemens_groen=(0/255,153/255,153/255)
o7=(255/255, 193/255, 158/255)
o1=(242/255, 98/255, 0/255)
siemens_groen_light1=(92/255,176/255,175/255)
siemens_groen_light2=(142/255,198/255,197/255)
siemens_groen_light3=(188/255,221/255,220/255)
siemens_blauw=(1/255,8/255,46/255)
siemens_blauw_groen=(0/255,90/255,120/255)
wit=(1,1,1)


# %% Upload aggregated data
loc_501_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated//loc_501_F6C.csv')#,index_col=0)
loc_501_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501_F18C.csv')#,index_col=0)

loc_531_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531_F6C.csv')#,index_col=0)
loc_531_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531_F18C.csv')#,index_col=0)

loc_528_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528_F6C.csv')#,index_col=0)
loc_528_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528_F18C.csv')#,index_col=0)

loc_502_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_502_F6C.csv')#,index_col=0)


# remove extra column
del loc_501_F6C['Unnamed: 0']
del loc_501_F18C['Unnamed: 0']

del loc_531_F6C['Unnamed: 0']
del loc_531_F18C['Unnamed: 0']
del loc_528_F6C['Unnamed: 0']
del loc_528_F18C['Unnamed: 0']

del loc_502_F6C['Unnamed: 0']

def fix_start_datum(data):
    data_all_datum=pd.to_datetime(data['start_datum']).dt.date
    data_all_datum2=pd.to_datetime(data_all_datum)
    data['start_datum'] =data_all_datum2

fix_start_datum(loc_501_F6C)
fix_start_datum(loc_501_F18C)    

fix_start_datum(loc_531_F6C)
fix_start_datum(loc_531_F18C)    

fix_start_datum(loc_528_F6C)
fix_start_datum(loc_528_F18C)   

fix_start_datum(loc_502_F6C)

# %% Set up data for location 501 and 531

# STEP 1: Add two lanes together
loc_501=loc_501_F6C.copy(deep=True)
loc_501['waarnemingen_intensiteit']=loc_501_F6C['waarnemingen_intensiteit']+loc_501_F18C['waarnemingen_intensiteit']

loc_531=loc_531_F6C.copy(deep=True)
loc_531['waarnemingen_intensiteit']=loc_531_F6C['waarnemingen_intensiteit']+loc_531_F18C['waarnemingen_intensiteit']

loc_528=loc_528_F6C.copy(deep=True)
loc_528['waarnemingen_intensiteit']=loc_528_F6C['waarnemingen_intensiteit']+loc_528_F18C['waarnemingen_intensiteit']

loc_502=loc_502_F6C.copy(deep=True)

#   C. Split in train and test set, do not look at test anymore from now on!
#   Chosen to split in voor half juli en na half juli zodat je alle soorten dagen (globaal) in zowel test als train set hebt

condition=loc_501['start_datum']<'2019-07-16'
loc_501_train=loc_501.loc[loc_501.index[condition],:]
loc_501_test=loc_501.loc[loc_501.index[-condition],:]   #negative condition works

condition=loc_531['start_datum']<'2019-07-16'
loc_531_train=loc_531.loc[loc_531.index[condition],:]
loc_531_test=loc_531.loc[loc_531.index[-condition],:]   #negative condition works

condition=loc_528['start_datum']<'2019-07-16'
loc_528_train=loc_528.loc[loc_528.index[condition],:]
loc_528_test=loc_528.loc[loc_528.index[-condition],:]   #negative condition works

condition=loc_502['start_datum']<'2019-07-16'
loc_502_train=loc_502.loc[loc_502.index[condition],:]
loc_502_test=loc_502.loc[loc_502.index[-condition],:]   #negative condition works

loc_501_train_stand_normal=stats.zscore(loc_501_train['waarnemingen_intensiteit']).reshape(loc_501_train.shape[0],)
loc_501_train_stand_normal_df=loc_501_train.copy()
loc_501_train_stand_normal_df['waarnemingen_intensiteit']=loc_501_train_stand_normal

# %% Meerdere dingen om te onderzoeken
# 1 normale autocorrelatie
# 2 median traffic flow eraf
# 3 daily median traffic flow eraf
my_ticks_one_week=['0','1','2','3','4','5','6','7']
my_ticks_two_weeks=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']
my_ticks_three_weeks=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21']
# 1. NORMAAL

# 1. Set up feature space. Eigenlijk wil ik y 24 keer verplaatsen
def normal_correlation(data,look_back,loc,ylim,ticks,auto_type):
    y_nu=data['waarnemingen_intensiteit']
    data_array=np.transpose(np.tile(y_nu,(look_back,1)))
    y_normal=pd.DataFrame(data_array)
    
    i=0
    for column in y_normal:
        y_normal[column]=y_normal[column].shift(i)
        i+=1
    
    # 2. Remove rows with NAN
    y_normal=y_normal.iloc[(look_back-1):] #remove first 48 rows to exclude nan values
    
    # 3. Find correlations 
    
    # A. linear correlation: Pearson
    corr_matrix_pear= y_normal.corr()
    cor_lin_values=corr_matrix_pear[0]#.sort_values(ascending=False)
    
    # B. non linear, spearmans rank
    corr_matrix_spear=y_normal.corr(method='spearman')
    cor_non_lin_values=corr_matrix_spear[0]#.sort_values(ascending=False))
    
    #my_ticks_normal=np.linspace(0,int(lookback/24),int(lookback/24)+1).tolist()
    #my_ticks_normal=list(np.linspace(0,int(lookback/24),int(lookback/24)+1))

    fig,ax =plt.subplots()
    plt.plot(cor_lin_values,marker='',label='Pearson',color=siemens_blauw,markersize=4)
    plt.plot(cor_non_lin_values,marker='',label='Spearman',color=siemens_groen,markersize=4)
    plt.xlabel('Time lag [days]')
    plt.title('Auto-correlation in traffic flow at location {}'.format(loc))
    plt.ylabel('Correlation coefficient')
    plt.legend(loc=1)
    plt.ylim(ylim)
    ax.set_xticks(np.linspace(0,lookback,len(ticks)))
    ax.set_xticklabels(ticks)
    plt.savefig(r"C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\figures\autocorr_{}{}.pdf".format(auto_type,loc),bbox_inches='tight')
    return y_normal
    
lookback=21*24
ylim=(-1,1.1)
y_normal_501=normal_correlation(loc_501_train,lookback,'501',ylim,my_ticks_three_weeks,'normal')
y_normal_531=normal_correlation(loc_531_train,lookback,'531',ylim,my_ticks_three_weeks,'normal')

#y_normal_528=normal_correlation(loc_528_train,lookback,'528',ylim,my_ticks_three_weeks,'normal')
#y_normal_502=normal_correlation(loc_502_train,lookback,'502',ylim,my_ticks_three_weeks,'normal')

#y_normal_501=normal_correlation(loc_501_train_stand_normal_df,lookback,'501',ylim,my_ticks_three_weeks,'normal_stand')


#%% MEDIAN ERAF

# 1. Find median and subtract
def deseason_hour(data): 
    data_clus=data.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
    data_median=data_clus.median()
    
    number_tile=int(data['waarnemingen_intensiteit'].shape[0]/24)
    data_median_full=np.transpose(np.tile(data_median,number_tile))
    deseason=pd.Series(data['waarnemingen_intensiteit'].array-data_median_full)
    deseason_df=pd.DataFrame({'waarnemingen_intensiteit':deseason})
    return deseason_df

loc_501_deseason=deseason_hour(loc_501_train)
loc_531_deseason=deseason_hour(loc_531_train)
loc_528_deseason=deseason_hour(loc_528_train)
loc_502_deseason=deseason_hour(loc_502_train)

loc_501_deseason_stand=loc_501_train.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
loc_501_deseason_stand=stats.zscore(loc_501_deseason_stand).flatten()
loc_501_deseason_stand_df=pd.DataFrame({'waarnemingen_intensiteit':loc_501_deseason_stand})

loc_531_deseason_stand=loc_531_train.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
loc_531_deseason_stand=stats.zscore(loc_531_deseason_stand).flatten()
loc_531_deseason_stand_df=pd.DataFrame({'waarnemingen_intensiteit':loc_531_deseason_stand})


lookback=21*24
ylim=(-0.25,1.1)
#%%
y_median_531_stand=normal_correlation(loc_531_deseason_stand_df,lookback,'531',ylim,my_ticks_three_weeks,'median_stand')
y_median_501_stand=normal_correlation(loc_501_deseason_stand_df,lookback,'501',ylim,my_ticks_three_weeks,'median_stand')
#%%
ylim=(-0.25,1.1)
y_median_501=normal_correlation(loc_501_deseason,lookback,'501',ylim,my_ticks_three_weeks,'median')
y_median_531=normal_correlation(loc_531_deseason,lookback,'531',ylim,my_ticks_three_weeks,'median')
#y_median_528=normal_correlation(loc_528_deseason,lookback,'528',ylim,my_ticks_three_weeks,'median')
#y_median_502=normal_correlation(loc_528_deseason,lookback,'502',ylim,my_ticks_three_weeks,'median')


#%% DOW MEDIAN

# Step 1 calculate dow median
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
    
    
    data_0_av=data_0.groupby(["start_tijd"]).median().reset_index(drop="False")['waarnemingen_intensiteit']
    data_1_av=data_1.groupby(["start_tijd"]).median().reset_index(drop="False")['waarnemingen_intensiteit']
    data_2_av=data_2.groupby(["start_tijd"]).median().reset_index(drop="False")['waarnemingen_intensiteit']
    data_3_av=data_3.groupby(["start_tijd"]).median().reset_index(drop="False")['waarnemingen_intensiteit']
    data_4_av=data_4.groupby(["start_tijd"]).median().reset_index(drop="False")['waarnemingen_intensiteit']
    data_5_av=data_5.groupby(["start_tijd"]).median().reset_index(drop="False")['waarnemingen_intensiteit']
    data_6_av=data_6.groupby(["start_tijd"]).median().reset_index(drop="False")  ['waarnemingen_intensiteit']  

    week_median=pd.concat([data_0_av,data_1_av,data_2_av,data_3_av,data_4_av,data_5_av,data_6_av])
    week_between=pd.Series(np.transpose(np.tile(week_median,(52*3))))
    week_total=pd.concat([data_6_av,week_between,data_0_av,data_1_av]).reset_index(drop=True)
    
    return week_total

week_total_501=median_dow(loc_501_train)
week_total_531=median_dow(loc_531_train)
week_total_528=median_dow(loc_528_train)
week_total_502=median_dow(loc_502_train)

# Haal dagen eruit die verwijderd zijn.
original_index_501=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_501.pkl")
week_median_501=week_total_501[original_index_501['original_index']].reset_index(drop=True)

original_index_531=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_531.pkl")
week_median_531=week_total_531[original_index_531['original_index']].reset_index(drop=True)

original_index_528=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_528.pkl")
week_median_528=week_total_528[original_index_528['original_index']].reset_index(drop=True)

original_index_502=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_502.pkl")
week_median_502=week_total_502[original_index_502['original_index']].reset_index(drop=True)

# deze is nog niet de train test set maar de totale
deseason_daily_501=pd.Series(loc_501_train['waarnemingen_intensiteit'].array-week_median_501[0:loc_501_train.shape[0]])
deseason_daily_501_df=pd.DataFrame({'waarnemingen_intensiteit':deseason_daily_501})
deseason_daily_531=pd.Series(loc_531_train['waarnemingen_intensiteit'].array-week_median_531[0:loc_531_train.shape[0]])
deseason_daily_531_df=pd.DataFrame({'waarnemingen_intensiteit':deseason_daily_531})
deseason_daily_528=pd.Series(loc_528_train['waarnemingen_intensiteit'].array-week_median_528[0:loc_528_train.shape[0]])
deseason_daily_528_df=pd.DataFrame({'waarnemingen_intensiteit':deseason_daily_528})
deseason_daily_502=pd.Series(loc_502_train['waarnemingen_intensiteit'].array-week_median_502[0:loc_502_train.shape[0]])
deseason_daily_502_df=pd.DataFrame({'waarnemingen_intensiteit':deseason_daily_502})

lookback=21*24
ylim=(-0.1,1.1)
y_median_day_501=normal_correlation(deseason_daily_501_df,lookback,'501',ylim,my_ticks_three_weeks,'median_day')
y_median_day_531=normal_correlation(deseason_daily_531_df,lookback,'531',ylim,my_ticks_three_weeks,'median_day')
#y_median_day_528=normal_correlation(deseason_daily_528_df,lookback,'528',ylim,my_ticks_three_weeks,'median_day')
#y_median_day_502=normal_correlation(deseason_daily_502_df,lookback,'502',ylim,my_ticks_three_weeks,'median_day')


# %% wat voor correlatie is deze correlatie? Maak een scatter_matrix die de verschillende features tegenover elkaar zet.
from pandas.plotting import scatter_matrix

for i in range(y_normal_501.shape[1]):
    scatter_matrix(y_normal_501[[0,i]],alpha=0.3,diagonal='hist',hist_kwds={'bins': 60})
    
#%%
for i in range(y_median_501.shape[1]):
    scatter_matrix(y_median_501[[0,i]],alpha=0.3,diagonal='hist',hist_kwds={'bins': 60})
    
for i in range(100):#y_median_day_501.shape[1]):
    scatter_matrix(y_median_day_501[[0,i]],alpha=0.3,diagonal='hist',hist_kwds={'bins': 60},color=o1)

#%% for location 531
for i in range(100):#y_normal_531.shape[1]):
    scatter_matrix(y_normal_531[[0,i]],alpha=0.3,diagonal='hist',hist_kwds={'bins': 60})
    
#%%
for i in range(100):#y_median_531.shape[1]):
    scatter_matrix(y_median_531[[0,i]],alpha=0.3,diagonal='hist',hist_kwds={'bins': 60})
    
#%%
for i in range(200):#y_median_day_531.shape[1]):
    scatter_matrix(y_median_day_531[[0,i]],alpha=0.3,diagonal='hist',hist_kwds={'bins': 60})
    
 