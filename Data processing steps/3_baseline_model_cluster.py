# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:47:43 2021

@author: z0049unj
"""
#Try clustering again, with scaled features and lane 1 and lane 2 together as feature

# %% Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import medfilt
from datetime import datetime
import seaborn as sns

# clustering
import calplot
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

from scipy import stats # for z-score normalization

# machine learning performance metrics
from sklearn.metrics import mean_squared_error

#Define
color_tu=(0, 166/255, 214/255)
#plt.rcParams["font.family"] = "sans-serif"
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
# %%
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


colors=[b3,b7,o3,o7]
colors_3=[b3,b7,o3]
colors_2=[b3,b7]
colors_7=[b1,b3,b7,b10,o1,o3,o7]


cmap_blue_orange_7 = LinearSegmentedColormap.from_list("", colors_7)

cmap_blue_orange_4 = LinearSegmentedColormap.from_list("", colors)
cmap_blue_orange_3 = LinearSegmentedColormap.from_list("", colors_3)
cmap_blue_orange_2 = LinearSegmentedColormap.from_list("", colors_2)

# %% Upload aggregated data

loc_501_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated//loc_501_F6C.csv')#,index_col=0)
loc_501_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501_F18C.csv')#,index_col=0)
loc_501r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501r_F6C.csv')#,index_col=0)
loc_501r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_501r_F18C.csv')#,index_col=0)

loc_502_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_502_F6C.csv')#,index_col=0)
loc_502r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_502r_F6C.csv')#,index_col=0)

loc_528_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528_F6C.csv')#,index_col=0)
loc_528_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528_F18C.csv')#,index_col=0)
loc_528r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528r_F6C.csv')#,index_col=0)
loc_528r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_528r_F18C.csv')#,index_col=0)

loc_530_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530_F6C.csv')#,index_col=0)
loc_530_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530_F18C.csv')#,index_col=0)
loc_530r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530r_F6C.csv')#,index_col=0)
loc_530r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_530r_F18C.csv')#,index_col=0)

loc_531_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531_F6C.csv')#,index_col=0)
loc_531_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531_F18C.csv')#,index_col=0)
loc_531r_F6C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531r_F6C.csv')#,index_col=0)
loc_531r_F18C=pd.read_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Preprocess_without_inserting_median/data_aggregated/loc_531r_F18C.csv')#,index_col=0)

# remove extra column
del loc_501_F6C['Unnamed: 0']
del loc_501_F18C['Unnamed: 0']
del loc_501r_F6C['Unnamed: 0']
del loc_501r_F18C['Unnamed: 0']

del loc_502_F6C['Unnamed: 0']
del loc_502r_F6C['Unnamed: 0']

del loc_528_F6C['Unnamed: 0']
del loc_528_F18C['Unnamed: 0']
del loc_528r_F6C['Unnamed: 0']
del loc_528r_F18C['Unnamed: 0']

del loc_530_F6C['Unnamed: 0']
del loc_530_F18C['Unnamed: 0']
del loc_530r_F6C['Unnamed: 0']
del loc_530r_F18C['Unnamed: 0']

del loc_531_F6C['Unnamed: 0']
del loc_531_F18C['Unnamed: 0']
del loc_531r_F6C['Unnamed: 0']
del loc_531r_F18C['Unnamed: 0']

def fix_start_datum(data):
    data_all_datum=pd.to_datetime(data['start_datum']).dt.date
    data_all_datum2=pd.to_datetime(data_all_datum)
    data['start_datum'] =data_all_datum2

fix_start_datum(loc_501_F6C)
fix_start_datum(loc_501_F18C)    
fix_start_datum(loc_501r_F6C)    
fix_start_datum(loc_501r_F18C)    

fix_start_datum(loc_502_F6C)
fix_start_datum(loc_502r_F6C)    

fix_start_datum(loc_528_F6C)
fix_start_datum(loc_528_F18C)    
fix_start_datum(loc_528r_F6C)    
fix_start_datum(loc_528r_F18C)  

fix_start_datum(loc_530_F6C)
fix_start_datum(loc_530_F18C)    
fix_start_datum(loc_530r_F6C)    
fix_start_datum(loc_530r_F18C)  

fix_start_datum(loc_531_F6C)
fix_start_datum(loc_531_F18C)    
fix_start_datum(loc_531r_F6C)    
fix_start_datum(loc_531r_F18C)  

# %% Set up data only for location 501 now


#1. scale feature waarnemingen intensiteit WIL JE NIET WANT DIT IS JE OUTPUT MAAKT NIET UIT ALS GROOT
# z-score standardization
loc_501_F6C_scaled=stats.zscore(loc_501_F6C['waarnemingen_intensiteit']) #zscore calucalted with mean and std of the entire set
loc_501_F18C_scaled=stats.zscore(loc_501_F18C['waarnemingen_intensiteit'])

#min max normalization
loc_501_F6C_scaled=(loc_501_F6C['waarnemingen_intensiteit']-loc_501_F6C['waarnemingen_intensiteit'].min())/(loc_501_F6C['waarnemingen_intensiteit'].max()-loc_501_F6C['waarnemingen_intensiteit'].min()) #min max normalization
loc_501_F18C_scaled=(loc_501_F18C['waarnemingen_intensiteit']-loc_501_F18C['waarnemingen_intensiteit'].min())/(loc_501_F18C['waarnemingen_intensiteit'].max()-loc_501_F18C['waarnemingen_intensiteit'].min())

loc_501_F6C['intensiteit_scaled']=loc_501_F6C_scaled
loc_501_F18C['intensiteit_scaled']=loc_501_F18C_scaled

# --> DAAROM DEZE UITGECOMMEND EN MET NORMALE
#X_501_F6C=loc_501_F6C.pivot_table(columns='start_tijd',values='intensiteit_scaled',index='start_datum')
#X_501_F18C=loc_501_F18C.pivot_table(columns='start_tijd',values='intensiteit_scaled',index='start_datum')

X_501_F6C=loc_501_F6C.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
X_501_F18C=loc_501_F18C.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')


X_501_clus=pd.concat([X_501_F6C,X_501_F18C],axis=1)

# info:
loc_501_F6C_mean=loc_501_F6C['waarnemingen_intensiteit'].mean()
loc_501_F18C_mean=loc_501_F18C['waarnemingen_intensiteit'].mean()
loc_501_F6C_std=loc_501_F6C['waarnemingen_intensiteit'].std()
loc_501_F18C_std=loc_501_F18C['waarnemingen_intensiteit'].std()

# daily mean
loc_501_F6C_daily_mean=X_501_F6C.mean()
loc_501_F6C_daily_median=X_501_F6C.median()
loc_501_F18C_daily_mean=X_501_F18C.mean()
loc_501_F18C_daily_median=X_501_F18C.median()

# %% Clustering 
# Hierarchical clustering

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(linkage='ward',distance_threshold=0, n_clusters=None).fit(X_501_clus)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# %% Clear split at 2,4,5,8
hierarchical_test=AgglomerativeClustering(n_clusters=2).fit(X_501_clus)
labels_hierar=hierarchical_test.labels_
print(np.count_nonzero(labels_hierar==0))       #65
print(np.count_nonzero(labels_hierar==1))       #290

# %%

def plot_clusters(data,N_clusters):
    hierarchical=AgglomerativeClustering(n_clusters=N_clusters).fit(data)
    labels_hierar=hierarchical.labels_+1 #such that cluster 1 and 2 cluster 0 are the missing days
    
    data_copy=data.copy(deep=True)
    data_copy['label']=labels_hierar
    calplot.calplot(data_copy['label'],textfiller='-') #hier nice kleur map toevoegen maar ligt aan hoeveel clusters
    
    plt.figure()
    plt.title('Number of clusters equals {}'.format(N_clusters))
    plt.subplot(1,2,1)
    plt.xlabel('time[h]')
    plt.ylabel('waarnemingen')
    plt.title('Sensor F6C')

    print(10*'*')
    print('number of clusters: ',N_clusters)
    for i in range(1,N_clusters+1): #0 niet want dat zijn de missing days
        print('amount in cluster',i,np.count_nonzero(labels_hierar==i))
        center=data_copy.loc[data_copy.index[data_copy['label']==i],:].mean()
        plt.plot(center[0:24],label=i)
    plt.legend()    
 
    plt.subplot(1,2,2)
    plt.xlabel('time[h]')
    plt.ylabel('waarnemingen')
    plt.title('Sensor F18C')
    for i in range(1,N_clusters+1):
        center=data_copy.loc[data_copy.index[data_copy['label']==i],:].mean()
        plt.plot(center[24:48],label=i)
    plt.legend()    
    return(data_copy)


    
X_501_2=plot_clusters(X_501_clus,2)
X_501_3=plot_clusters(X_501_clus,3)
X_501_4=plot_clusters(X_501_clus,4)
X_501_5=plot_clusters(X_501_clus,5)
X_501_6=plot_clusters(X_501_clus,6)
X_501_7=plot_clusters(X_501_clus,7)
X_501_8=plot_clusters(X_501_clus,8)
X_501_9=plot_clusters(X_501_clus,9)
X_501_10=plot_clusters(X_501_clus,10)

# %% eigenlijk 2 main clusters en dan each wat outliers op de zondagen, feestdagen, vakantiedagen, hier heb ik nog geen features van
#dus al die extra dingen worden geen features en worden niet meegenomen
# baseline predictor, jan feb mar, oct nov dec cluster 4 (als gesplitst in 4 clusters)
                    # apr,mei,jun,jul,aug, sep    


# number of clusters =4, cluster 2 en cluster 4  , zomer en winter respectively  
condition=X_501_4['label']==2          #summer
X_501_summer_F6C_z=X_501_4.loc[X_501_4.index[condition],:].mean()[0:24]
X_501_summer_F18C_z=X_501_4.loc[X_501_4.index[condition],:].mean()[24:48]

condition=X_501_4['label']==4           #winter
X_501_winter_F6C_z=X_501_4.loc[X_501_4.index[condition],:].mean()[0:24]
X_501_winter_F18C_z=X_501_4.loc[X_501_4.index[condition],:].mean()[24:48]

# reproduce actual data back from z-score
X_501_summer_F6C=X_501_summer_F6C_z*loc_501_F6C_std+loc_501_F6C_mean
X_501_summer_F18C=X_501_summer_F18C_z*loc_501_F18C_std+loc_501_F18C_mean
X_501_winter_F6C=X_501_winter_F6C_z*loc_501_F6C_std+loc_501_F6C_mean
X_501_winter_F18C=X_501_winter_F18C_z*loc_501_F18C_std+loc_501_F18C_mean


fig,(ax1,ax2)=plt.subplots(2,1,sharey=True)
ax1.plot(X_501_summer_F6C,label='summer')
ax1.plot(X_501_winter_F6C,label='winter')
ax1.set(xlabel='time', ylabel='waarnemingen intensiteit [/h]')
ax1.title.set_text('F6C')
ax1.legend()

ax2.plot(X_501_summer_F18C,label='summer')
ax2.plot(X_501_winter_F18C,label='winter')
ax2.set(xlabel='time', ylabel='waarnemingen intensiteit [/h]')
ax2.title.set_text('F18C')
ax2.legend()



# %% Set features bewaar, hier zijn ze geschaald

# 1 scale features eigenlijk moet elk uur apart gescaled worden voor clusteren
        # al gedaan hierboven bij z-score normalization, een mean en std per locatie per laan gebruikt, niet per uur ofzo
y1_501=loc_501_F6C['intensiteit_scaled']
y2_501=loc_501_F18C['intensiteit_scaled']

# 2 add other time axis 
    # sine and cosine over the time, such that model knows time is cyclical https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/

#Step A, make time hour, represent by integer 0-23    
loc_501_F6C['hour'] = pd.to_datetime(loc_501_F6C['start_tijd'], format='%H:%M:%S').dt.hour   
hours_in_day=24

sin_time=np.sin(2*np.pi*loc_501_F6C['hour']/hours_in_day) 
cos_time=np.cos(2*np.pi*loc_501_F6C['hour']/hours_in_day) 

# Make Feature space
# 1. prediction y1 y2
y_501=pd.DataFrame({'F6C':y1_501,'F18C':y2_501})

# 2. features t1 t2 
X_501=pd.DataFrame({'sin_time':sin_time,'cos_time':cos_time,'start_datum':loc_501_F6C['start_datum']})
loc_501=pd.concat([y_501,X_501],axis=1)

loc_501_train= loc_501.sample(frac=0.8, random_state=0)
loc_501_test = loc_501.drop(loc_501_train.index)

y_501_train=loc_501_train[['F6C','F18C']]
X_501_train=loc_501_train[['sin_time','cos_time']]
y_501_test=loc_501_test[['F6C','F18C']]
X_501_test=loc_501_test[['sin_time','cos_time']]

# %% predict with baseline 
'''
prediction_baseline=X_501_4.copy(deep=True)
prediction_baseline.drop('label',inplace=True,axis=1)
X_501_winter_total=np.concatenate((X_501_winter_F6C ,X_501_winter_F18C))
X_501_summer_total=np.concatenate((X_501_summer_F6C ,X_501_summer_F18C))

condition=prediction_baseline.index<'2019-04-01'
prediction_baseline.loc[prediction_baseline.index[condition],:]=X_501_winter_total
condition=prediction_baseline.index>='2019-04-01'
prediction_baseline.loc[prediction_baseline.index[condition],:]=X_501_summer_total
condition=prediction_baseline.index>'2019-09-30'
prediction_baseline.loc[prediction_baseline.index[condition],:]=X_501_winter_total
'''


#%% FIRST MODEL: LINEAR REGRESSION
#predict output: Loc_501_F6C, loc_501_F18C
#features: tijd, seizoen (winter zomer of soort sinus), dow, holiday (pagina 67 one hot encoding in python)

# STEP 1 MAKE FEATURE SPACE
#   A. make time hour, represent by integer 0-23    
loc_501_F6C['hour'] = pd.to_datetime(loc_501_F6C['start_tijd'], format='%H:%M:%S').dt.hour   
hours_in_day=24

sin_time=np.sin(2*np.pi*loc_501_F6C['hour']/hours_in_day) 
cos_time=np.cos(2*np.pi*loc_501_F6C['hour']/hours_in_day) 

#   B. make dataframe time, ouput, datum
# 2. features t1 t2 
loc_501=pd.DataFrame({'sin_time':sin_time,'cos_time':cos_time,'start_datum':loc_501_F6C['start_datum'],'F6C':loc_501_F6C['waarnemingen_intensiteit'],'F18C':loc_501_F18C['waarnemingen_intensiteit'],'weekday':loc_501_F6C['weekday'],'start_tijd':loc_501_F6C['start_tijd']})

#   C. Split in train and test set, do not look at test anymore from now on!
def split_train_test(data,test_ratio):
    np.random.seed(42)          #such that the same test/train split is made everytime, not if dataset is updated, then something else has to be thought of
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

#from sklearn.model_selection import train_test_split   can be usefull does the same as the above, can also insert multiple dataframes which it will split at the same indices
loc_501_train,loc_501_test=split_train_test(loc_501,0.2)

# %% STEP 2 FEATURE SCALING
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler #for z-score standardization numerical features
from sklearn.preprocessing import OneHotEncoder  # for transformation of categorical features
from sklearn.compose import ColumnTransformer #to transform both numerical and categorical features simultaneously
'''
#   A. Separate in numerical and categorical variables
# chosen not to scale time variables, because already between -1 and 1, due to train test split else not the cos sin values relative to eachother
#loc_501_num= now empty because i do not want to scale my output
loc_501_num=[]
loc_501_cat=loc_501_train['weekday']

#   B. Scale features, numerical values with z-score, categorical to one hot encoding
num_attributes=[]       # the column names empty
cat_attributes=['weekday']

Scale_data =ColumnTransformer([
    ('num',StandardScaler(),num_attributes),
    ('cat',OneHotEncoder(),cat_attributes),
    ])

scaled_test_full=Scale_data.fit_transform(loc_501_train)
print(scaled_test_full)
'''
''' HIER BEN IK NOG NIET IK HOEF NOG NIET TE SCHALEN '''
#     A. split features and output nu puur tijd
y_501_train=loc_501_train[['F6C','F18C']]
X_501_train=loc_501_train[['sin_time','cos_time']]
y_501_test=loc_501_test[['F6C','F18C']]
X_501_test=loc_501_test[['sin_time','cos_time']]

# %% Start with linear model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X_501_train,y_501_train)

# lets try on a few instances of the training set
print('prediction:',lin_reg.predict(X_501_train[:5]))
print('true:',y_501_train[:5])

time_one_day=np.transpose(np.array([sin_time[0:24],cos_time[0:24]]))
lin_pred_one_day=lin_reg.predict(time_one_day)
print('prediction:',lin_reg.predict(time_one_day))

plt.figure()
plt.subplot(1,2,1)
plt.plot(lin_pred_one_day[:,0],label='predicted')
plt.plot(loc_501_F6C_daily_mean,label='mean')
plt.plot(loc_501_F6C_daily_median,label='median')
plt.title('F6C')
plt.legend()

plt.subplot(1,2,2)
plt.plot(lin_pred_one_day[:,1],label='predicted')
plt.plot(loc_501_F18C_daily_mean,label='mean')
plt.plot(loc_501_F18C_daily_median,label='median')
plt.title('F18C')
plt.legend()

from sklearn.metrics import mean_squared_error
lin_501_predict=lin_reg.predict(X_501_train)
lin_mse=mean_squared_error(y_501_train,lin_501_predict)
lin_rmse=np.sqrt(lin_mse)
print(lin_rmse)

# %% probeer inzicht te krijgen in de fouten
error=y_501_train-lin_501_predict
error['start_tijd']=loc_501_train['start_tijd']
error['start_datum']=loc_501_train['start_datum']
error['weekday']=loc_501_train['weekday']

# groupby start_tijd en laat error verdeling zien
plt.figure()
plt.subplot(1,2,1)
plt.scatter(error['start_tijd'],error['F6C'],label='error F6C',alpha=0.1)
plt.legend()

plt.subplot(1,2,2)
plt.scatter(error['start_tijd'],error['F18C'],label='error F18C',alpha=0.1)
plt.legend()

# kijk welke dagen grootste error is, alleen niet echt fair denk ik want onduidelijk wat allemaal in training set zit
# ik kan wel kijken op welke dagen de grootste errors zijn, bij F6C >100, bij F18C >250
# deze errors zijn groter dan 0 omdat meerdere meetpunten per dag, je ziet dat alles meer dan 15, super veel, betekend dus dat op elke dag waar de meeste fouten zijn meer dan 15 meetpunten dat hebben
# veel van deze errors op dezelfde dagen bij de twee banen.
error=error.set_index('start_datum')
error['label_F6C']=0
condition=error['F6C']>100
error.loc[error.index[condition],'label_F6C']=1

error['label_F18C']=0
condition=error['F18C']>250
error.loc[error.index[condition],'label_F18C']=1

calplot.calplot(error['label_F6C'])
calplot.calplot(error['label_F18C'])

#%% MODEL 2: DECISION TREE
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(X_501_train,y_501_train)

#evaluate on training set
tree_501_predict=tree_reg.predict(X_501_train)
tree_mse=mean_squared_error(y_501_train,tree_501_predict)
tree_rmse=np.sqrt(tree_mse)
print(tree_rmse) # al klein beetje beter

# %%visualize decision tree
from sklearn.tree import export_graphviz
from sklearn import tree 

text_rep=tree.export_text(tree_reg)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(tree_reg, 
                   feature_names=X_501_train.columns,  
                   filled=True)

# %%plot prediction --> eigenlijk gewoon de mean
tree_reg_pred_one_day=tree_reg.predict(time_one_day)
print('prediction:',tree_reg.predict(time_one_day))

plt.figure()
plt.subplot(1,2,1)
plt.plot(lin_pred_one_day[:,0],label='lin')
plt.plot(tree_reg_pred_one_day[:,0],label='tree')

plt.plot(loc_501_F6C_daily_mean,label='mean')
plt.plot(loc_501_F6C_daily_median,label='median')
plt.title('F6C')
plt.legend()

plt.subplot(1,2,2)
plt.plot(lin_pred_one_day[:,1],label='lin')
plt.plot(tree_reg_pred_one_day[:,1],label='tree ')

plt.plot(loc_501_F18C_daily_mean,label='mean')
plt.plot(loc_501_F18C_daily_median,label='median')
plt.title('F18C')
plt.legend()




# %% MODEL 3: RANDOM FOREST 
# je kan de importance van elke feature plotten.

from sklearn.ensemble import RandomForestRegressor
forest_reg=RandomForestRegressor(n_estimators=100,n_jobs=-1) #n_estimators number of trees, n_jobs=1 normaal. -1means using all processors (heb ik nu nog niet)
forest_reg.fit(X_501_train,y_501_train)

#evaluate on training set
forest_501_predict=forest_reg.predict(X_501_train)
forest_mse=mean_squared_error(y_501_train,forest_501_predict)
forest_rmse=np.sqrt(forest_mse)
print(forest_rmse) #slechter

for name,score in zip(X_501_train.columns,forest_reg.feature_importances_):
    print(name,score)

#oke interessant cosines time is veel belangrijker, ik denk om welk stuk van de dag het gaat dat cosinus niet twee keer dezelfde waardes heeft en sin toevallig wel


# %% MODEL 4: SUPPORT VECTOR MACHINE
# kan zowel een linear als nonlinear model maken
from sklearn.svm import LinearSVR
from sklearn.svm import SVR

from sklearn.multioutput import MultiOutputRegressor #required if multiple outputs

#LINEAR
#single output regressor
svm_reg=LinearSVR(epsilon=2) #epsilon is groot de margin is, hyperparameter

# Create the Multioutput Regressor
svm_mul_reg = MultiOutputRegressor(svm_reg)

svm_mul_reg.fit(X_501_train,y_501_train)

#evaluate on training set
svm_501_predict=svm_mul_reg.predict(X_501_train)
svm_mse=mean_squared_error(y_501_train,svm_501_predict)
svm_rmse=np.sqrt(svm_mse)
print('linear svm',svm_rmse) # slechter

# %%NON LINEAR
#single output regressor
svm_reg_nl=SVR(kernel="rbf",degree=1,epsilon=0.1) #linear, poly,rbf,sigmoid,precomputed opties voor kernel

# Create the Multioutput Regressor
svm_mul_reg_nl = MultiOutputRegressor(svm_reg_nl)

svm_mul_reg_nl.fit(X_501_train,y_501_train)

#evaluate on training set
svm_501_predict_nl=svm_mul_reg_nl.predict(X_501_train)
svm_mse_nl=mean_squared_error(y_501_train,svm_501_predict_nl)
svm_rmse_nl=np.sqrt(svm_mse_nl)
print('non linear svm',svm_rmse_nl) # nog veeeel slechter slechter


# %% PLot overything so far
svm_pred_one_day=svm_mul_reg.predict(time_one_day)
svm_nl_pred_one_day=svm_mul_reg_nl.predict(time_one_day)
forest_pred_one_day=forest_reg.predict(time_one_day)

plt.figure()
plt.subplot(1,2,1)
plt.plot(lin_pred_one_day[:,0],label='lin')
plt.plot(tree_reg_pred_one_day[:,0],label='tree')
plt.plot(svm_pred_one_day[:,0],label='lin svm')
plt.plot(svm_nl_pred_one_day[:,0],label='non lin svm')
plt.plot(forest_pred_one_day[:,0],label='forest')

plt.plot(loc_501_F6C_daily_mean,label='mean')
plt.plot(loc_501_F6C_daily_median,label='median')
plt.title('F6C')
plt.legend()

plt.subplot(1,2,2)
plt.plot(lin_pred_one_day[:,1],label='lin')
plt.plot(tree_reg_pred_one_day[:,1],label='tree ')
plt.plot(svm_pred_one_day[:,1],label='lin svm')
plt.plot(svm_nl_pred_one_day[:,1],label='non lin svm')
plt.plot(forest_pred_one_day[:,1],label='forest')

plt.plot(loc_501_F18C_daily_mean,label='mean')
plt.plot(loc_501_F18C_daily_median,label='median')
plt.title('F18C')
plt.legend()



# %%
# day of the week possibilities: 
    # 1 features weekday, weekend 
''' go to one hot encoding
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()'''
    # 2 ma-do, vr,za,zo
    # 3 ma,di,wo,do,vr,za,zo

# holiday 

# season possibilities:
    # 1 summer winter
    # continuous variable summer winter

# weather

# events