# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:59:29 2021

@author: z0049unj
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

def add_weekday(data_imported_jan,data_imported_feb,data_imported_maart,data_imported_april,data_imported_mei,data_imported_juni,data_imported_juli,data_imported_aug,data_imported_sep,data_imported_okt,data_imported_nov,data_imported_dec):
    data_imported_jan['start_datum'] = pd.to_datetime(data_imported_jan['start_datum'],dayfirst=True)
    data_imported_jan['weekday'] = data_imported_jan['start_datum'].dt.dayofweek
    data_imported_feb['start_datum'] = pd.to_datetime(data_imported_feb['start_datum'],dayfirst=True)
    data_imported_feb['weekday'] = data_imported_feb['start_datum'].dt.dayofweek
    data_imported_maart['start_datum'] = pd.to_datetime(data_imported_maart['start_datum'],dayfirst=True)
    data_imported_maart['weekday'] = data_imported_maart['start_datum'].dt.dayofweek
    data_imported_april['start_datum'] = pd.to_datetime(data_imported_april['start_datum'],dayfirst=True)
    data_imported_april['weekday'] = data_imported_april['start_datum'].dt.dayofweek
    data_imported_mei['start_datum'] = pd.to_datetime(data_imported_mei['start_datum'],dayfirst=True)
    data_imported_mei['weekday'] = data_imported_mei['start_datum'].dt.dayofweek
    data_imported_juni['start_datum'] = pd.to_datetime(data_imported_juni['start_datum'],dayfirst=True)
    data_imported_juni['weekday'] = data_imported_juni['start_datum'].dt.dayofweek
    data_imported_juli['start_datum'] = pd.to_datetime(data_imported_juli['start_datum'],dayfirst=True)
    data_imported_juli['weekday'] = data_imported_juli['start_datum'].dt.dayofweek
    data_imported_aug['start_datum'] = pd.to_datetime(data_imported_aug['start_datum'],dayfirst=True)
    data_imported_aug['weekday'] = data_imported_aug['start_datum'].dt.dayofweek
    data_imported_sep['start_datum'] = pd.to_datetime(data_imported_sep['start_datum'],dayfirst=True)
    data_imported_sep['weekday'] = data_imported_sep['start_datum'].dt.dayofweek
    data_imported_okt['start_datum'] = pd.to_datetime(data_imported_okt['start_datum'],dayfirst=True)
    data_imported_okt['weekday'] = data_imported_okt['start_datum'].dt.dayofweek
    data_imported_nov['start_datum'] = pd.to_datetime(data_imported_nov['start_datum'],dayfirst=True)
    data_imported_nov['weekday'] = data_imported_nov['start_datum'].dt.dayofweek
    data_imported_dec['start_datum'] = pd.to_datetime(data_imported_dec['start_datum'],dayfirst=True)
    data_imported_dec['weekday'] = data_imported_dec['start_datum'].dt.dayofweek

    # Different function for 2019 and 2018/2017 omdat bij 2019 steeds een extra dag eraan zit en bij de andere twee niet
def separate_data_on_location_2019(name1,name2,name3,name4,name5,name6,name7,name8,name9,name10,data):
    data.drop("gebruikte_minuten_intensiteit", axis=1, inplace=True)    
    data.drop('gebruikte_minuten_snelheid', axis=1, inplace=True) 
    data.drop('data_error_snelheid', axis=1, inplace=True) 
    data.drop('data_error_intensiteit', axis=1, inplace=True) 
    data.drop('gem_intensiteit', axis=1, inplace=True) 
    data.drop('gewogen_gem_snelheid', axis=1, inplace=True) 


    grouped_data=data.groupby(data["id_meetlocatie"])
    loc_1=grouped_data.get_group(name1)
    loc_1=loc_1.reset_index(drop="True")
    loc_1.drop(index=loc_1.index[-288*12:],axis=0,inplace=True)
    
    loc_2=grouped_data.get_group(name2)
    loc_2=loc_2.reset_index(drop="True")
    loc_2.drop(index=loc_2.index[-288*12:],axis=0,inplace=True)

    loc_3=grouped_data.get_group(name3)
    loc_3=loc_3.reset_index(drop="True")
    loc_3.drop(index=loc_3.index[-288*6:],axis=0,inplace=True) #CAREFUL different because 6 sensors instead of 12

    loc_4=grouped_data.get_group(name4)
    loc_4=loc_4.reset_index(drop="True")
    loc_4.drop(index=loc_4.index[-288*6:],axis=0,inplace=True)

    loc_5=grouped_data.get_group(name5)
    loc_5=loc_5.reset_index(drop="True")
    loc_5.drop(index=loc_5.index[-3456:],axis=0,inplace=True)

    loc_6=grouped_data.get_group(name6)
    loc_6=loc_6.reset_index(drop="True")
    loc_6.drop(index=loc_6.index[-3456:],axis=0,inplace=True)

    loc_7=grouped_data.get_group(name7)
    loc_7=loc_7.reset_index(drop="True")
    loc_7.drop(index=loc_7.index[-3456:],axis=0,inplace=True)

    loc_8=grouped_data.get_group(name8)
    loc_8=loc_8.reset_index(drop="True")
    loc_8.drop(index=loc_8.index[-3456:],axis=0,inplace=True)

    loc_9=grouped_data.get_group(name9)
    loc_9=loc_9.reset_index(drop="True")
    loc_9.drop(index=loc_9.index[-3456:],axis=0,inplace=True)

    loc_10=grouped_data.get_group(name10)
    loc_10=loc_10.reset_index(drop="True")
    loc_10.drop(index=loc_10.index[-3456:],axis=0,inplace=True)

    return loc_1,loc_2,loc_3,loc_4,loc_5,loc_6,loc_7,loc_8,loc_9,loc_10

def separate_data_on_location_2018_2017(name1,name2,name3,name4,name5,name6,name7,name8,name9,name10,data):
    data.drop("gebruikte_minuten_intensiteit", axis=1, inplace=True)    
    data.drop('gebruikte_minuten_snelheid', axis=1, inplace=True) 
    data.drop('data_error_snelheid', axis=1, inplace=True) 
    data.drop('data_error_intensiteit', axis=1, inplace=True) 
    data.drop('gem_intensiteit', axis=1, inplace=True) 
    data.drop('gewogen_gem_snelheid', axis=1, inplace=True) 


    grouped_data=data.groupby(data["id_meetlocatie"])
    loc_1=grouped_data.get_group(name1)
    loc_1=loc_1.reset_index(drop="True")
    
    loc_2=grouped_data.get_group(name2)
    loc_2=loc_2.reset_index(drop="True")

    loc_3=grouped_data.get_group(name3)
    loc_3=loc_3.reset_index(drop="True")

    loc_4=grouped_data.get_group(name4)
    loc_4=loc_4.reset_index(drop="True")

    loc_5=grouped_data.get_group(name5)
    loc_5=loc_5.reset_index(drop="True")

    loc_6=grouped_data.get_group(name6)
    loc_6=loc_6.reset_index(drop="True")

    loc_7=grouped_data.get_group(name7)
    loc_7=loc_7.reset_index(drop="True")

    loc_8=grouped_data.get_group(name8)
    loc_8=loc_8.reset_index(drop="True")

    loc_9=grouped_data.get_group(name9)
    loc_9=loc_9.reset_index(drop="True")

    loc_10=grouped_data.get_group(name10)
    loc_10=loc_10.reset_index(drop="True")

    return loc_1,loc_2,loc_3,loc_4,loc_5,loc_6,loc_7,loc_8,loc_9,loc_10

def extract_2lane_sensor_data(data):
    data_grouped=data.groupby(data["ndw_index"])                            #group data based on index
    data_F6C= data_grouped.get_group("F6C")                                 #extract values with index F6C
    data_F18C=data_grouped.get_group("F18C") 
    data_F6C= data_F6C.reset_index(drop="True")                             #reset index dataframe     
    data_F18C= data_F18C.reset_index(drop="True")                           #reset index dataframe         
    return data_F6C , data_F18C
     
def extract_502_sensor_data(data):
    data_grouped=data.groupby(data["ndw_index"])                            #group data based on index
    data_F6C= data_grouped.get_group("F6C")                                 #extract values with index F6C
    data_F6C= data_F6C.reset_index(drop="True")                             #reset index dataframe     
    return data_F6C

def rows_with_nan(data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18):
    index1 = pd.isnull(data1).any(1).to_numpy().nonzero()[0].tolist()
    index2 = pd.isnull(data2).any(1).to_numpy().nonzero()[0].tolist()
    index3 = pd.isnull(data3).any(1).to_numpy().nonzero()[0].tolist()
    index4 = pd.isnull(data4).any(1).to_numpy().nonzero()[0].tolist()

    index5 = pd.isnull(data5).any(1).to_numpy().nonzero()[0].tolist()
    index6 = pd.isnull(data6).any(1).to_numpy().nonzero()[0].tolist()

    index7 = pd.isnull(data7).any(1).to_numpy().nonzero()[0].tolist()
    index8 = pd.isnull(data8).any(1).to_numpy().nonzero()[0].tolist()
    index9 = pd.isnull(data9).any(1).to_numpy().nonzero()[0].tolist()
    index10 = pd.isnull(data10).any(1).to_numpy().nonzero()[0].tolist()

    index11 = pd.isnull(data11).any(1).to_numpy().nonzero()[0].tolist()
    index12 = pd.isnull(data12).any(1).to_numpy().nonzero()[0].tolist()
    index13 = pd.isnull(data13).any(1).to_numpy().nonzero()[0].tolist()
    index14 = pd.isnull(data14).any(1).to_numpy().nonzero()[0].tolist()


    index15 = pd.isnull(data15).any(1).to_numpy().nonzero()[0].tolist()
    index16 = pd.isnull(data16).any(1).to_numpy().nonzero()[0].tolist()
    index17 = pd.isnull(data17).any(1).to_numpy().nonzero()[0].tolist()
    index18 = pd.isnull(data18).any(1).to_numpy().nonzero()[0].tolist()
    
    return index1,index2,index3,index4,index5,index6,index7,index8,index9,index10,index11,index12,index13,index14,index15,index16,index17,index18


def preprocess_interpoleer(data,rows): #data loc_501_F6C, rows is rows_with_nan_501_6
    # declare
    loc_datum_u=data['start_datum'][rows].unique()
    loc_datum=data['start_datum'][rows]
    dataframe = pd.Series(loc_datum,name="start_datum").to_frame()

    # Investigate missing values for each day

    for i in range(loc_datum_u.shape[0]):  
        day_grouped=dataframe.groupby(dataframe["start_datum"])
        day_index=day_grouped.get_group(loc_datum_u[i]) #index
        #find out how many points consequtive order
        conseq=np.ones(day_index.shape[0],dtype=int)                         #array met 1 en zo groot als aantal missing values op de desbetreffende dag
        interval=0                                                           #in het hoeveelste slot per dag zit je
        
        #ga conseq invullen met hoeveel achterelkaar volgende indexen
        for k in range(day_index.shape[0]-1):
            if day_index.index[k] == day_index.index[k+1]-1:
                conseq[interval]+=1
            else:
                interval+=1
        

        #obtain indices each "cluster"
        length=[]
        condition=0
        
        for j in range(interval+1):
            if condition>0:
                break
            
            if j==0:                        #first cluster
                length.append(conseq[j])        #make vector with the size of each cluster
                index_x1=day_index.index[0]
                index_x2=day_index.index[length[0]-1]
                if conseq[j]<=3:
                    print('all fine',loc_datum_u[i])

                else:
                    if data['start_tijd'][index_x1]>'00:00:55' and data['start_tijd'][index_x2]<'04:00:00':
                        #1. extract all the same days
                        loc_dow=data.groupby(["weekday"])
                        day_data=loc_dow.get_group(data['weekday'][index_x1])
                        day_data=day_data.reset_index(drop="True")
                        day_median=day_data.groupby(["start_tijd"]).median().reset_index(drop="False")
                        start_tijd_dag=data['start_tijd'][0:288]
                        day_median['start_tijd']=start_tijd_dag                         # add collumn start_tijd

                        
                        for d in range(length[j]): # voor elk gemiste punt j                             
                            start_tijd=data['start_tijd'][index_x1+j]

                            waar_int=day_median.loc[day_median['start_tijd']==start_tijd,'waarnemingen_intensiteit'].item()
                            gem_snel=day_median.loc[day_median['start_tijd']==start_tijd,'gem_snelheid']
                            data.at[index_x1+d,'waarnemingen_intensiteit']=waar_int 
                            data.at[index_x1+d,'gem_snelheid']=gem_snel 
                            data.at[index_x1+d,'waarnemingen_snelheid']=waar_int 
                        
                        
                    else:
                        indexes_day=data[data['start_datum']==loc_datum_u[i]].index #corresponding indexes
                        data.drop(index=indexes_day,inplace=True) #drop

                        condition=1
                        #data.drop(index=indexes_day,inplace=True) #drop
                    
            else:
                index_x1=day_index.index[sum(length)]
                length.append(conseq[j])        #make vector with the size of each cluster
                index_x2=day_index.index[sum(length)-1]  
                if conseq[j]<=3:
                    print('all fine')

                else:
                    if data['start_tijd'][index_x1]>'00:00:55' and data['start_tijd'][index_x2]<'04:00:00':
                        #1. extract all the same days
                        loc_dow=data.groupby(["weekday"])
                        day_data=loc_dow.get_group(data['weekday'][index_x1])
                        day_data=day_data.reset_index(drop="True")
                        day_median=day_data.groupby(["start_tijd"]).median().reset_index(drop="False")
                        start_tijd_dag=data['start_tijd'][0:288]
                        day_median['start_tijd']=start_tijd_dag
                        # add collumn start_tijd
                        for d in range(length[j]): # voor elk gemiste punt j                             
                            start_tijd=data['start_tijd'][index_x1+j]
                            waar_int=day_median.loc[day_median['start_tijd']==start_tijd,'waarnemingen_intensiteit'].item()
                            gem_snel=day_median.loc[day_median['start_tijd']==start_tijd,'gem_snelheid']
                            data.at[index_x1+d,'waarnemingen_intensiteit']=waar_int 
                            data.at[index_x1+d,'gem_snelheid']=gem_snel 
                            data.at[index_x1+d,'waarnemingen_snelheid']=waar_int   

                            
                    else:
                        indexes_day=data[data['start_datum']==loc_datum_u[i]].index #corresponding indexes
                        data.drop(index=indexes_day,inplace=True) #drop
                        condition=1
def preprocess_missing_vel(data):
    errors_neg_value=0
    errors_no_velocity=0
    errors_no_intensity=0
    median=[]   #to check median or locf
    
    
    #First calculat for FC6
    for i in data.index:

        if data['waarnemingen_intensiteit'][i]<0 or data['waarnemingen_snelheid'][i]<0:     
            #print("Negative value for intensity or velocity, please look at the data at",i)
            data.at[i,'waarnemingen_intensiteit']=0
            data.at[i,'waarnemingen_snelheid']=0            
            data.at[i,'gem_snelheid']=-1            
            data.at[i,'gem_intensiteit']=0              
            errors_neg_value+=1
            
        elif data['waarnemingen_intensiteit'][i]>0 and data['gem_snelheid'][i]==-1:
            data.at[i,"waarnemingen_intensiteit"]=0
            data.at[i,'waarnemingen_snelheid']=0
            
        elif data['waarnemingen_intensiteit'][i]>0 and data['waarnemingen_snelheid'][i]==0:
           # print("intensiteit meting geen snelheid meting, please look at the data at ",i)
            errors_no_velocity+=1
            if 12<=i<data.shape[0]-11: 

                median_new=data[i-12:i+11].median()['gem_snelheid']             
                median.append(median_new)
              
                #Reset at the appropriate index in the total data
                data.at[i,"waarnemingen_snelheid"]=data['waarnemingen_intensiteit'][i]
                data.at[i,'gem_snelheid']=median_new 

            elif 0<i<12:
                median_new=data['gem_snelheid'][0:12].max()      #median is altijd -1 dat wil je niet want dan heb je nog steeds geen snelheid
                median.append(median_new)

                data.at[i,"waarnemingen_snelheid"]=data['waarnemingen_intensiteit'][i]
                data.at[i,'gem_snelheid']=median_new 
                
            elif i==0:
                data.at[i,"waarnemingen_snelheid"]=data['waarnemingen_intensiteit'][i]
                data.at[i,'gem_snelheid']=data["gem_snelheid"][i+1]
                print('CAREFUL CHECK i=0')
                
            elif data.shape[0]-11<=i<data.shape[0]-1:                
                median_new=data[i-(data.shape[0]-i):data.shape[0]]['gem_snelheid'].max()
                median.append(median_new)
                
                data.at[i,"waarnemingen_snelheid"]=data['waarnemingen_intensiteit'][i]
                data.at[i,'gem_snelheid']=median_new
                
                
            elif i==data.shape[0]-1:
                data.at[i,"waarnemingen_snelheid"]=data['waarnemingen_intensiteit'][i]
                data.at[i,'gem_snelheid']=data["gem_snelheid"][i-1]
                print('CAREFUL CHECK {}'.format(i))
                
                
        elif data['waarnemingen_intensiteit'][i]==0 and data['waarnemingen_snelheid'][i]>0:    
            #print("geen intensiteit meting wel snelheid meting ")
            errors_no_intensity+=1
    return errors_neg_value,errors_no_velocity,errors_no_intensity, median

def plot_median_vel(data):
    grouped=data.groupby(['start_tijd']).describe(percentiles=[.05,.25, .5, .75,.95])
    
    median = grouped['gem_snelheid', '50%']
    median.name = 'gem_snelheid'
    quartiles1 = grouped['gem_snelheid', '25%']
    quartiles3 = grouped['gem_snelheid', '75%']
    q_5= grouped['gem_snelheid', '5%']
    q_95= grouped['gem_snelheid', '95%']
    x = grouped.index
        
    ax =plt.figure()
    ax=sns.lineplot(x=x, y=median,color=b1,label='median')
    ax.fill_between(x, q_5, q_95,color=b6,label='5%-95%'); 
    ax.fill_between(x, quartiles1, quartiles3,color=b3,label='25%-75%')
    plt.legend()
    plt.title('Statistics velocity Location {} lane {}'.format(data['id_meetlocatie'][0],data['ndw_index'][0])) 
    
def filter_outlier_int_theoretical(data,max_intensiteit):
    condition=data['waarnemingen_intensiteit']>max_intensiteit
    loc_filtered=data.loc[data.index[condition],:]
    
    plt.figure()
    plt.plot(data['waarnemingen_intensiteit'],color=siemens_groen_light2,label='Real data')
    plt.scatter(loc_filtered.index,loc_filtered['waarnemingen_intensiteit'],color=siemens_blauw,label='Filtered points')
    plt.title('Traffic flow in 2019 at location and lane ')
    
    #change waarnemingen_intensiteit and gem_snelheid at these locations to nan
    data.loc[loc_filtered.index,'waarnemingen_intensiteit']=np.nan
    data.loc[loc_filtered.index,'gem_snelheid']=np.nan
   
    plt.plot(data['waarnemingen_intensiteit'],color=siemens_groen,label='Filtered data')
    plt.legend()
    plt.ylabel('Traffic flow [vehicles]')
    plt.xlabel('Time [5 min]')
    plt.savefig("test2.svg")
    
    plt.show()
    
    return loc_filtered

def filter_outlier_int_kernel(data): #based on statistics output indices that are outside
    #extract 1 dow
    index_total=[]
    for i in range(7):
        #for each day of the week, since profile differs between days
        loc_dow=data.groupby(['weekday'])
        day_data=loc_dow.get_group(i)

        #nu door naar statistics boxplot
        data_stats = day_data.groupby(['start_tijd']).describe(percentiles=[.05,.25, .5, .75,.95])
        median = data_stats['waarnemingen_intensiteit', '50%']
        median.name = 'waarnemingen_intensiteit'
        quartiles1 = data_stats['waarnemingen_intensiteit', '25%']
        quartiles3 = data_stats['waarnemingen_intensiteit', '75%']
        q_5= data_stats['waarnemingen_intensiteit', '5%']
        q_95= data_stats['waarnemingen_intensiteit', '95%']
        x = data_stats.index
        '''
        #UNCOMMENT IF DESIRED TO PLOT
        ax =plt.figure()
        ax=sns.lineplot(x=x, y=median,color=siemens_blauw,label='median')
        ax.fill_between(x, q_5, q_95,color=siemens_groen_light2,label='5%-95%'); 
        ax.fill_between(x, quartiles1, quartiles3,color=siemens_groen,label='25%-75%')
        plt.legend()
       # plt.title('Statistics Location {} and lane {} at Monday'.format(location,data,)) 
        plt.xticks(rotation=80)
        plt.savefig("medianstats.svg")
        plt.xlabel('Time [hour of the day]')
        plt.ylabel('Traffic flow [vehicles]')
        '''
       
    
        datum_unique=day_data['start_datum'][:].unique()
        q_5=q_5.reset_index()
        q_95=q_95.reset_index()
        index=[]
         
        for j in range(datum_unique.shape[0]): 
            data_unique_1=day_data.groupby(day_data['start_datum']).get_group(datum_unique[j]) #ik wil index niet resetten want dan niet duidelijk welke dag, daarom moet index in q_5 en q_95 veranderd worden
            q_5=q_5.set_index(data_unique_1.index[:])
            q_95=q_95.set_index(data_unique_1.index[:])  
            condition_low=data_unique_1['waarnemingen_intensiteit']<q_5['waarnemingen_intensiteit','5%'] 
            condition_up=data_unique_1['waarnemingen_intensiteit']>q_95['waarnemingen_intensiteit','95%']  
            index.extend(data_unique_1.index[condition_low | condition_up])

        index_total.extend(index) 
    index_total.sort()
    return index_total

def add_kernel(data_F6C,data_F18C,kernel,index_total_F6C,index_total_F18C):   
    data_copy_F6C=data_F6C.copy(deep=True)                                                  #deep true (default) such that changes in copy do not show in original    
    data_kernel_F6C=data_copy_F6C.rolling(kernel,min_periods=1,center=True).median()        #set minperiods to 1, else first and last values will become Nan due to window size
    data_copy_F6C['kernel_intensiteit']=data_kernel_F6C['waarnemingen_intensiteit']
    
    data_copy_F18C=data_F18C.copy(deep=True)                                                #deep true (default) such that changes in copy do not show in original
    data_kernel_F18C=data_copy_F18C.rolling(kernel,min_periods=1,center=True).median()
    data_copy_F18C['kernel_intensiteit']=data_kernel_F18C['waarnemingen_intensiteit']
        
    condition1_F6C=data_copy_F6C['waarnemingen_intensiteit']>3*data_copy_F6C['kernel_intensiteit']
    data_filtered1_F6C=data_copy_F6C.loc[data_copy_F6C.index[condition1_F6C],:]
    
    condition1_F18C=data_copy_F18C['waarnemingen_intensiteit']>3*data_copy_F18C['kernel_intensiteit']
    data_filtered1_F18C=data_copy_F18C.loc[data_copy_F18C.index[condition1_F18C],:]

    condition2_F6C=data_filtered1_F6C['waarnemingen_intensiteit']>50
    condition2_F18C=data_filtered1_F18C['waarnemingen_intensiteit']>50

    data_filtered2_F6C=data_filtered1_F6C.loc[data_filtered1_F6C.index[condition2_F6C],:]
    data_filtered2_F18C=data_filtered1_F18C.loc[data_filtered1_F18C.index[condition2_F18C],:]
    
    condition3_F6C=data_filtered2_F6C.index.isin(index_total_F6C)
    condition3_F18C=data_filtered2_F18C.index.isin(index_total_F18C)
    
    data_final_F6C=data_filtered2_F6C.loc[data_filtered2_F6C.index[condition3_F6C]]
    data_final_F18C=data_filtered2_F18C.loc[data_filtered2_F18C.index[condition3_F18C]]
    
    return data_final_F6C,data_final_F18C

def remove_outliers(data,data_final):
    data.loc[data_final.index,'waarnemingen_intensiteit']=np.nan                #different than previous because nan not removed yet so indices coincide
    data.loc[data_final.index,'gem_snelheid']=np.nan


def filter_outlier_velocity(data,max_vel):
    condition_vel=data['gem_snelheid']>max_vel
    data_too_fast=data.loc[data.index[condition_vel],:]
    mediaan=data.describe()['gem_snelheid']['50%']

    
    print(data.loc[data_too_fast.index[:],'gem_snelheid'])
    data.at[data_too_fast.index[:],'gem_snelheid']=mediaan
    print(data.loc[data_too_fast.index[:],'gem_snelheid'])
    
    return data_too_fast   

def delete_columns(data):
    # 1. Remove unneccessary columns    
    del data['eind_meetperiode'], data['waarnemingen_snelheid'], data['rijstrook_rijbaan'], data['voertuigcategorie'], data['Unnamed: 0']

    # 2. Delete weekday because aggregation will go wrong otherwise, add again at the end
    del data['weekday']    
    
def aggregate(data,rows):

    # 1. Add a new column to the dataframe, 0 if normal and 1 if nan,required to adjust sum intensity if nan values   
    extra_info=np.zeros(len(data))
    data['info_nan']=extra_info
    data.loc[rows,'info_nan']=1

    # 4. Make one column with day and time in timestring such that we can resample
    new_column_data=pd.to_datetime(data['start_datum'] + ' ' + data['start_tijd'])
    data['timestamp']=new_column_data
    data['snelheid']=data['waarnemingen_intensiteit']*data['gem_snelheid']      #data['snelheid] is required to calculate average velocity

    #to know which days have been removed during preprocessing
    data_start_datum=pd.to_datetime(data['start_datum'].unique())
    data['start_tijd']=pd.to_datetime(data['start_tijd'])

    #aggregate into hour time intervals and add start_datum and start_tijd again
    data=data.set_index('timestamp').resample('H').sum().reset_index()
    data_all_datum=pd.to_datetime(data['timestamp']).dt.date
    data_all_datum2=pd.to_datetime(data_all_datum)
    data['start_datum'] =data_all_datum2

    data_tijd=pd.to_datetime(data['timestamp']).dt.time
    data['start_tijd'] =data_tijd
    
    #fix intensity due to missing values
    data_waarnemingen_scaled=data['waarnemingen_intensiteit']*(1+data['info_nan']/10)
    data['waarnemingen_intensiteit']=data_waarnemingen_scaled
    
    #fix velocity , divide by number of cars
    real_velocity=data['snelheid']/data['waarnemingen_intensiteit']
    data['gem_snelheid']=real_velocity

    #remove sum velocity
    del data['snelheid'], data['info_nan'], data['timestamp']
    
    #remove days again, all values are 0 
    data = data[data['start_datum'].isin(data_start_datum)]

    #Add weekday
    data['weekday'] = data['start_datum'].dt.dayofweek
    return data 
   