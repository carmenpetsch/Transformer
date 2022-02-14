# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 14:05:42 2021

@author: z0049unj
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 09:47:43 2021

@author: z0049unj
"""


# %% Import packages
import pandas as pd
import numpy as np

# to plot figures
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# clustering
import calplot
from sklearn.cluster import AgglomerativeClustering
from scipy import stats # for z-score normalization

# import functions from baseline_models_functions
from baseline_models_functions import plot_dendrogram
from baseline_models_functions import plot_clusters
from baseline_models_functions import linear_model
from baseline_models_functions import plot_errors_on_day
from baseline_models_functions import decision_tree_model
from baseline_models_functions import random_forest_model
from baseline_models_functions import svm_linear
from baseline_models_functions import svm_nonlinear

# machine learning models
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import tree 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# machine learning performance metrics

# %% Define colors
color_tu=(0, 166/255, 214/255)
#plt.rcParams["font.family"] = "sans-serif"
wit=(1,1,1)
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

colors_2=[b3,b7]
colors_3=[b3,b7,o3]
colors_4=[b3,b7,o3,o7]
# colors 5 en 6 nodig voor calender plots, voeg wit toe om daarmee de dagen te doen die niet bestaan
colors_5=[wit,o1,o7,b1,b7]
colors_6=[wit,b1,b7,o1,o4,o7]
colors_7=[b1,b3,b7,b10,o1,o3,o7]

cmap_blue_orange_2 = LinearSegmentedColormap.from_list("", colors_2)
cmap_blue_orange_3 = LinearSegmentedColormap.from_list("", colors_3)
cmap_blue_orange_4 = LinearSegmentedColormap.from_list("", colors_4)
cmap_blue_orange_5 = LinearSegmentedColormap.from_list("", colors_5)
cmap_blue_orange_6 = LinearSegmentedColormap.from_list("", colors_6)
cmap_blue_orange_7 = LinearSegmentedColormap.from_list("", colors_7)
cmap_blue_orange_9 = LinearSegmentedColormap.from_list("", [wit,b1,b3,b7,b10,o1,o3,o7,o10])

# %% Upload aggregated data and lanes added
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


# %% Upload feature sets
y_501_train=pd.read_pickle("Dataframes_features/y_501_train.pkl")
y_501_test=pd.read_pickle("Dataframes_features/y_501_test.pkl")

X_501_train_time=pd.read_pickle("Dataframes_features/X_501_time.pkl")
X_501_test_time=pd.read_pickle("Dataframes_features/X_501_time_test.pkl")

X_501_train_dow=pd.read_pickle("Dataframes_features/X_501_dow.pkl")
X_501_test_dow=pd.read_pickle("Dataframes_features/X_501_dow_test.pkl")

X_501_train_season=pd.read_pickle("Dataframes_features/X_501_season.pkl")
X_501_test_season=pd.read_pickle("Dataframes_features/X_501_season_test.pkl")

X_501_train_feestdag=pd.read_pickle("Dataframes_features/X_501_feestdag.pkl")
X_501_test_feestdag=pd.read_pickle("Dataframes_features/X_501_feestdag_test.pkl")

y_501_train_nuldim=np.ravel(y_501_train,order='C')

condition_train=loc_501['start_datum']<'2019-06-01'
loc_501_train=loc_501.loc[loc_501.index[condition_train],:]
loc_501_train['sin_time']=X_501_train_time['sin_time']
loc_501_train['cos_time']=X_501_train_time['cos_time']


#%% originial index before removing days with nan
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
# %% Data preparation
# uiteindelijk niet want dit is je output, wel om te kijken naar dagen clusteren
# maakt niet uit want nu maar een feature, zelfde resultaat als wel en niet geschaald. Bij andere bestand niet omdat daar twee lanen als verschillende features werden gezien.

# z-score standardization
loc_501_scaled=stats.zscore(loc_501['waarnemingen_intensiteit']) #zscore calucalted with mean and std of the entire set

#min max normalization
#loc_501_scaled=(loc_501['waarnemingen_intensiteit']-loc_501['waarnemingen_intensiteit'].min())/(loc_501['waarnemingen_intensiteit'].max()-loc_501['waarnemingen_intensiteit'].min()) #min max normalization
#Deze maakt niets uit want nu schaal je de hele traffic flow als een feature maar hieronder schaal je de traffic flow per uur als een feature
#loc_501['intensiteit_scaled']=loc_501_scaled

# Uncomment als je geschaalde feature wilt
#X_501_clus=loc_501.pivot_table(columns='start_tijd',values='intensiteit_scaled',index='start_datum')



#Uncomment als je niet geschaalde features wilt
X_501_clus=loc_501.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_501r_clus=loc_501r.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_528_clus=loc_528.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_528r_clus=loc_528r.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_502_clus=loc_502.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_502r_clus=loc_502r.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_530_clus=loc_530.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_530r_clus=loc_530r.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
X_531_clus=loc_531.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
#X_531r_clus=loc_531r.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')

# Kan ook geschaald makt niet uit
X_501_clus_Scaled=X_501_clus.apply(stats.zscore)
X_531_clus_Scaled=X_531_clus.apply(stats.zscore)

# info:
loc_501_mean=loc_501['waarnemingen_intensiteit'].mean()
loc_501_std=loc_501['waarnemingen_intensiteit'].std()

# daily mean
loc_501_daily_mean=X_501_clus.mean()
loc_501_daily_median=X_501_clus.median()


# %% Hierarchical clustering
from baseline_models_functions import plot_dendrogram
from scipy.cluster import hierarchy

def hierarchical_clustering(Data,loc):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(linkage='ward',distance_threshold=0, n_clusters=None).fit(Data)
    
    # plot the top three levels of the dendrogram
    # set color paletter
    
    hierarchy.set_link_color_palette(['#019999','#5CB0AF', '#8EC6C5','#BCDDDC','#01082E'])
    hierarchy.set_link_color_palette(['#5CB0AF', '#BCDDDC'])
    
    plot_dendrogram(model, 'level', 4,'501')
    plt.savefig("dendrogram_501.svg")
    
    # Clear split at 3 4 and 5
    hierarchical_test=AgglomerativeClustering(n_clusters=2).fit(Data)
    labels_hierar=hierarchical_test.labels_
    print(np.count_nonzero(labels_hierar==0))       #75
    print(np.count_nonzero(labels_hierar==1))       #280
    
    # plot clusters 3 en 5 en 7 deze duidelijk uit dendogram
    siemens_blauw_tussen=[0,89/255,121/255]
    siemens_blauw_groen=(0/255,90/255,120/255)

    colors_3=[wit,siemens_groen,siemens_groen_light3,o1]
    cmap_3= LinearSegmentedColormap.from_list("", colors_3)
    X_501_3=plot_clusters(Data,3,cmap_3,colors_3,loc,'3')
    
    
    colors_5=[wit,siemens_groen,siemens_blauw_tussen,o1,o5,siemens_groen_light3]#,siemens_groen_light3]
    cmap_5= LinearSegmentedColormap.from_list("", colors_5)
    X_501_5=plot_clusters(Data,5,cmap_5,colors_5,loc,'5')
    
    colors_7=[wit,siemens_blauw_tussen,siemens_groen_light3,siemens_groen,o5,o9,siemens_blauw,o1]
    cmap_7= LinearSegmentedColormap.from_list("", colors_7)
    X_501_7=plot_clusters(Data,7,cmap_7,colors_7,loc,'7')
    

    return X_501_5


X_501_5=hierarchical_clustering(X_501_clus,'501')
X_501_5=hierarchical_clustering(X_501_clus_Scaled,'501_scaled')
#%%
X_531_5=hierarchical_clustering(X_531_clus,'531')
X_531_5=hierarchical_clustering(X_531_clus_Scaled,'531_scaled')


#%% Also investigate location 531
def hierarchical_clustering(Data,loc):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(linkage='ward',distance_threshold=0, n_clusters=None).fit(Data)
    
    # plot the top three levels of the dendrogram
    # set color paletter
    
    hierarchy.set_link_color_palette(['#019999','#5CB0AF', '#8EC6C5','#BCDDDC','#01082E'])
    hierarchy.set_link_color_palette(['#5CB0AF', '#BCDDDC'])
    
    plot_dendrogram(model, 'level', 4,'531')
    
    plt.savefig("dendrogram_531.svg")
    
    # Clear split at 3 4 and 5
    hierarchical_test=AgglomerativeClustering(n_clusters=2).fit(Data)
    labels_hierar=hierarchical_test.labels_
    print(np.count_nonzero(labels_hierar==0))       #75
    print(np.count_nonzero(labels_hierar==1))       #280
    
    # plot clusters 3 en 5 en 7 deze duidelijk uit dendogram
    siemens_blauw_tussen=[0,89/255,121/255]
    siemens_blauw_groen=(0/255,90/255,120/255)
    '''
    colors_5=[wit,siemens_groen,siemens_blauw_tussen,o1,siemens_groen_light3,o5]#,siemens_groen_light3]
    cmap_5= LinearSegmentedColormap.from_list("", colors_5)
    X_531_5=plot_clusters(Data,5,cmap_5,colors_5,loc,'5')
    '''
    colors_6=[wit,o1,siemens_blauw,siemens_blauw_tussen,siemens_groen,o5,siemens_groen_light3]#,siemens_groen_light3]
    cmap_6= LinearSegmentedColormap.from_list("", colors_6)
    X_531_6=plot_clusters(Data,6,cmap_6,colors_6,loc,'6')
    
    return X_531_5


X_531_5=hierarchical_clustering(X_531_clus,'531')

#%% Investigate March 2017, strange cluster
X_531_clus_part=X_531_clus.loc[X_531_clus.index<'2017-03-31',: ]#and X_531_clus.index>'2017-02-14' 
X_531_clus_part2=X_531_clus_part.loc[X_531_clus_part.index>'2017-02-20',: ].reset_index(drop=True)
#%%
for i in range(len(X_531_clus_part2)):
    plt.figure()
    plt.plot(X_531_clus_part2.loc[i,:])
    
#%%
X_502_5=hierarchical_clustering(X_502_clus,'502') # lijkt te veel op 501
#%%
X_528_5=hierarchical_clustering(X_528_clus,'528') # eigenllijk niet het andere uiterste

#%%
X_528r_5=hierarchical_clustering(X_528r_clus,'528r') # best wat dagen die missen en 3 hele gekke dagen, die moeten eruit haha
#%%
X_530_5=hierarchical_clustering(X_530_clus,'530') # helft van 2017 weg
#%%
X_530r_5=hierarchical_clustering(X_530r_clus,'530r') # helft van 2017 weg

#%%
X_531_5=hierarchical_clustering(X_531_clus,'531')

#%%
X_531r_5=hierarchical_clustering(X_531r_clus,'531r')

#%%
import seaborn as sns


def plot_cluster_stats(data,loc,cluster):
    X_501_5_cluster=data.loc[data.index[data['label']==cluster],:]
    del X_501_5_cluster['label']
    data_stats = X_501_5_cluster.describe(percentiles=[.05,.25, .5, .75,.95])
    
    
    median = data_stats.loc['50%',:]
    median.name = 'waarnemingen_intensiteit'
    
    quartiles1 = data_stats.loc['25%',:]
    quartiles3 = data_stats.loc['75%',:]
    q_5= data_stats.loc['5%',:]
    q_95= data_stats.loc['95%',:]
    x = data_stats.columns
    
    
    ax =plt.figure()
    ax=sns.lineplot(x=x, y=median,color=siemens_blauw,label='median')
    ax.fill_between(x, q_5, q_95,color=siemens_groen_light2,label='5%-95%'); 
    ax.fill_between(x, quartiles1, quartiles3,color=siemens_groen,label='25%-75%')
    plt.legend()
    plt.title('Cluster {} statistics for Location {} '.format(cluster,loc)) 
    plt.xticks(rotation=80)
    plt.savefig("cluster_{}_stats.svg".format(cluster))
    plt.xlabel('Time [hour of the day]')
    plt.ylabel('Traffic flow [vehicles]')
    
plot_cluster_stats(X_501_5,'501',1)   
plot_cluster_stats(X_501_5,'501',2)   
plot_cluster_stats(X_501_5,'501',3)   
plot_cluster_stats(X_501_5,'501',4)   
plot_cluster_stats(X_501_5,'501',5)   

plot_cluster_stats(X_531_5,'531',1)   
plot_cluster_stats(X_531_5,'531',2)   
plot_cluster_stats(X_531_5,'531',3)   
plot_cluster_stats(X_531_5,'531',4)   
plot_cluster_stats(X_531_5,'531',5)   


'''
X_501_4.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/X_501_4.csv')
X_501_5.to_csv('C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/X_501_5.csv')
'''

# %% ONLY TIME: X_501_train_time
# MODEL 1: LINEAR MODEL
lin_reg,lin_rmse,lin_rmse_cross,lin_501_predict=linear_model(X_501_train_time,y_501_train)
print('lin_rmse',lin_rmse)
print('lin_rmse_cross',lin_rmse_cross.mean())
# Plot the prediction made for one day
time_one_day=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24]]))
lin_pred_one_day=lin_reg.predict(time_one_day)

plt.figure()
plt.plot(lin_pred_one_day,color=colors_3[2],label='linear regression')
plt.plot(loc_501_daily_mean,color=colors_3[0],label='mean')
plt.plot(loc_501_daily_median,color=colors_3[1],label='median')
plt.title('Predicted number of observations')
plt.ylabel('Number of observations [1/h]')
plt.xlabel('Time')
plt.legend()
plt.xticks(rotation=80)

# probeer inzicht te krijgen in de fouten
# deze errors zijn groter dan 0 omdat meerdere meetpunten per dag, je ziet dat alles meer dan 15, super veel, betekend dus dat op elke dag waar de meeste fouten zijn meer dan 15 meetpunten dat hebben
plot_errors_on_day(loc_501_train, y_501_train, lin_501_predict,400)

y_total=y_501_train.copy(deep=True)
y_pred_all=lin_reg.predict(X_501_train_time)
y_total['lin_pred_time']=y_pred_all
rmse_lin_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['lin_pred_time'])**2)/y_total.shape[0])
mae_lin_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['lin_pred_time']))/y_total.shape[0]

print('linear rmse only time:',rmse_lin_time)
print('linear mae only time:',mae_lin_time)


#%% MODEL 2: DECISION TREE
# Classical model
max_depth=[]
min_samples_leaf=[]
tree_reg,tree_rmse,tree_rmse_cross,tree_501_predict=decision_tree_model(X_501_train_time,y_501_train,max_depth,min_samples_leaf)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict,400)


# Hyperparameter optimization
# hyperparameters: max_depth, min_samples_split,min_samples_leaf
# by GridsearchCv

'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9],'min_samples_leaf' : [1,2,3,4]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train,y_501_train)
grid_search.best_params_ #{'max_depth': 9, 'min_samples_leaf': 1}
grid_search.best_estimator_'''


tree_reg,tree_rmse,tree_rmse_cross,tree_501_predict=decision_tree_model(X_501_train_time,y_501_train,9,1)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict,400)

print('decision tree cross val score',tree_rmse_cross.mean())
'''# Visualize decision tree
text_rep=tree.export_text(grid_search.best_estimator_)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(grid_search.best_estimator_, 
                   feature_names=X_501_train.columns,  
                   filled=True)'''

# %%plot prediction --> eigenlijk gewoon de mean want je minimaliseerd mse
tree_reg_pred_one_day=tree_reg.predict(time_one_day)
print('prediction:',tree_reg.predict(time_one_day))

plt.figure()
plt.plot(lin_pred_one_day,color=o3,label='Linear regression')
plt.plot(tree_reg_pred_one_day,color=o7,label='Decision tree')

plt.plot(loc_501_daily_mean,color=b3,label='mean')
plt.plot(loc_501_daily_median,color=b7,label='median')
plt.title('Predicted number of observations')
plt.xlabel('Time')
plt.ylabel('Error [observations]')
plt.legend()
plt.xticks(rotation=80)

y_pred_dec=tree_reg.predict(X_501_train_time)
y_total['tree_pred_time']=y_pred_dec
rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

print('linear rmse only time:',rmse_tree_time)
print('linear mae only time:',mae_tree_time)


# %% MODEL 3: RANDOM FOREST 
# je kan de importance van elke feature plotten.
forest_reg,forest_rmse,forest_rmse_cross,forest_predict=random_forest_model(X_501_train_time,y_501_train,100,1,9,1)
print('random forest cross rmse',forest_rmse_cross.mean())
# Interessant cosines time is veel belangrijker, ik denk om welk stuk van de dag het gaat dat cosinus niet twee keer dezelfde waardes heeft en sin toevallig wel

# %% MODEL 4: SUPPORT VECTOR MACHINE
# kan zowel een linear als nonlinear model maken
svm_reg,svm_rmse,svm_predict=svm_linear(X_501_train_time,y_501_train,1)                              #linear model
svm_reg_nl,svm_rmse_nl,svm_predict_nl=svm_nonlinear(X_501_train_time,y_501_train,'poly',1,0.1)       # nonlinear model
'''
# hyperparameter optimization
param_grid=[{'kernel':['linear', 'poly', 'rbf','sigmoid' ],'degree' : [1,2,3,4,5,6,7],'epsilon':[0.1,0.5,1,1.5,2,2.5,3,3.5]}]

svm_reg_nl_h=SVR()

grid_search=GridSearchCV(svm_reg_nl_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train,y_501_train_nuldim)
grid_search.best_params_
grid_search.best_estimator_'''
'''
for i in range(30):
    eps=0.1+i*0.1
    svm_reg_nl=SVR(kernel="poly",degree=1,epsilon=eps)
    svm_reg_nl.fit(X_501_train,y_501_train_nuldim)

    #evaluate on training set
    svm_501_predict=svm_reg_nl.predict(X_501_train)
    svm_mse=mean_squared_error(y_501_train,svm_501_predict)
    svm_rmse=np.sqrt(svm_mse)
    print('linear svm with eps',eps,'RMSE:',svm_rmse) # slechter
'''

# %% Plot overything so far
svm_pred_one_day=svm_reg.predict(time_one_day)
svm_nl_pred_one_day=svm_reg_nl.predict(time_one_day)
forest_pred_one_day=forest_reg.predict(time_one_day)

plt.figure()
plt.plot(lin_pred_one_day,color=o1,label='Linear regression')
plt.plot(tree_reg_pred_one_day,color=o3,label='Decision tree')
plt.plot(svm_pred_one_day,color=b7,label='Linear SVM')
plt.plot(svm_nl_pred_one_day,color=b1,label='Non linear SVM')
plt.plot(forest_pred_one_day,color=o7,label='Random forest')

plt.plot(loc_501_daily_mean,color='k',linestyle='dashed',label='mean')
plt.plot(loc_501_daily_median,color='k',label='median')
plt.title('Predicted intensity')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)

# %% Set up features for DOW
# 1: DAY OF THE WEEK, POSSIBILITIES:
        # 1. weekday weekend (2 extra features)
        # 2. ma-do, vr,za,zo (4 extra features)
        # 3. all days of the week (7 extra features)

# OPTION NUMBER 1: weekday and weekend
# add weekday to the features
data_train=[loc_501_train['sin_time'],loc_501_train['cos_time'],loc_501_train['weekday']]
headers_train=['sin_time','cos_time','weekday']
X_501_train_weekday=pd.concat(data_train,axis=1,keys=headers_train)

X_train_days_opt1=X_501_train_weekday.copy(deep=True)
X_train_days_opt1.loc[:,'week']=0
X_train_days_opt1.loc[:,'weekend']=0

condition_weekend=X_train_days_opt1['weekday']>4 #then weekend
X_train_days_opt1.loc[X_train_days_opt1.index[condition_weekend],'weekend']=1
X_train_days_opt1.loc[X_train_days_opt1.index[-condition_weekend],'week']=1

X_train_days_opt1=X_train_days_opt1.drop('weekday',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# OPTION NUMBER 2: MA-DO, VR,ZA,ZO
X_501_train_day2= X_501_train_weekday.copy(deep=True)

X_501_train_day2.loc[:,'midweek']=0
X_501_train_day2.loc[:,'fri']=0
X_501_train_day2.loc[:,'sat']=0
X_501_train_day2.loc[:,'sun']=0
condition_midweek=X_501_train_day2['weekday']<4 #then midweek
condition_fri=X_501_train_day2['weekday']==4
condition_sat=X_501_train_day2['weekday']==5
condition_sun=X_501_train_day2['weekday']==6

X_501_train_day2.loc[X_501_train_day2.index[condition_midweek],'midweek']=1
X_501_train_day2.loc[X_501_train_day2.index[condition_fri],'fri']=1
X_501_train_day2.loc[X_501_train_day2.index[condition_sat],'sat']=1
X_501_train_day2.loc[X_501_train_day2.index[condition_sun],'sun']=1

X_501_train_day2=X_501_train_day2.drop('weekday',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# OPTION NUMBER 3: all days seperate
X_501_train_alld=X_501_train_weekday.copy(deep=True)
X_501_train_alld.loc[:,'mon']=0
X_501_train_alld.loc[:,'tue']=0
X_501_train_alld.loc[:,'wed']=0
X_501_train_alld.loc[:,'thu']=0
X_501_train_alld.loc[:,'fri']=0
X_501_train_alld.loc[:,'sat']=0
X_501_train_alld.loc[:,'sun']=0
condition_mon=X_501_train_alld['weekday']==0
condition_tue=X_501_train_alld['weekday']==1
condition_wed=X_501_train_alld['weekday']==2
condition_thu=X_501_train_alld['weekday']==3
condition_fri=X_501_train_alld['weekday']==4
condition_sat=X_501_train_alld['weekday']==5
condition_sun=X_501_train_alld['weekday']==6

X_501_train_alld.loc[X_501_train_alld.index[condition_mon],'mon']=1
X_501_train_alld.loc[X_501_train_alld.index[condition_tue],'tue']=1
X_501_train_alld.loc[X_501_train_alld.index[condition_wed],'wed']=1
X_501_train_alld.loc[X_501_train_alld.index[condition_thu],'thu']=1
X_501_train_alld.loc[X_501_train_alld.index[condition_fri],'fri']=1
X_501_train_alld.loc[X_501_train_alld.index[condition_sat],'sat']=1
X_501_train_alld.loc[X_501_train_alld.index[condition_sun],'sun']=1

X_501_train_alld=X_501_train_alld.drop('weekday',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# %%plot median van elke dag van de week zodat te zien wat wel en niet bij elkaar gegroepeerd wordt wss
def median_dow(data,location):
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

    plt.figure()
    plt.plot(data_0_av["waarnemingen_intensiteit"],color=o1,label='monday')
    plt.plot(data_1_av["waarnemingen_intensiteit"],color=o4,label='tuesday')
    plt.plot(data_2_av["waarnemingen_intensiteit"],color='k',label='wednesday')
    plt.plot(data_3_av["waarnemingen_intensiteit"],color=o8,label='thursday')
    plt.plot(data_4_av["waarnemingen_intensiteit"],color=b1,label='friday')
    plt.plot(data_5_av["waarnemingen_intensiteit"],color=b3,label='saturday')
    plt.plot(data_6_av["waarnemingen_intensiteit"],color=b7,label='sunday')
    plt.title('Median daily intensity at {}'.format(location))
    plt.xlabel('Time')
    plt.legend()
    plt.xticks(rotation=80)
    
median_dow(loc_501,'loc_501')
median_dow(loc_501r,'loc_501r')
median_dow(loc_502,'loc_502')
median_dow(loc_502r,'loc_502r')
median_dow(loc_528,'loc_528')
median_dow(loc_528r,'loc_528r')
median_dow(loc_530,'loc_530')
median_dow(loc_530r,'loc_530r')
median_dow(loc_531,'loc_531')
median_dow(loc_531r,'loc_531r')
# Check aswell for another location, ideally the model is generalized
#%% LINEAR MODEL

# OPTION 1
lin_reg_wd,lin_rmse_wd,lin_rmse_wd_cross,lin_501_predict_wd=linear_model(X_train_days_opt1,y_501_train)
print('lin_reg_wd cross rmse',lin_rmse_wd_cross.mean())
# Plot the two possible predictions
array_one=np.ones(24)
array_zero=np.zeros(24)
X_one_day_week=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_one,array_zero]))
X_one_day_weekend=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_one]))

lin_pred_week=lin_reg_wd.predict(X_one_day_week)
lin_pred_weekend=lin_reg_wd.predict(X_one_day_weekend)

plt.figure()
plt.plot(lin_pred_week,color=o1,label='linear predicted week')
plt.plot(lin_pred_weekend,color=o3,label='linear predicted weekend')
plt.plot(lin_pred_one_day,color=o5,label='linear predicted all days')

plt.plot(loc_501_daily_mean,color=b3,label='mean')
plt.plot(loc_501_daily_median,color=b7,label='median')
plt.title('Hourly number of observations')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)

# %% OPTION 2
lin_reg_midw,lin_rmse_midw,lin_rmse_midw_cross,lin_501_predict_midw=linear_model(X_501_train_day2,y_501_train)

# Plot the four possible predictions
array_one=np.ones(24)
array_zero=np.zeros(24)
X_one_day_midw=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_one,array_zero,array_zero,array_zero]))
X_one_day_fri_midw=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_one,array_zero,array_zero]))
X_one_day_sat_midw=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_one,array_zero]))
X_one_day_sun_midw=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_zero,array_one]))


lin_pred_midw=lin_reg_midw.predict(X_one_day_midw)
lin_pred_fri=lin_reg_midw.predict(X_one_day_fri_midw)
lin_pred_sat=lin_reg_midw.predict(X_one_day_sat_midw)
lin_pred_sun=lin_reg_midw.predict(X_one_day_sun_midw)

plt.figure()
plt.plot(lin_pred_midw,color=o1,label='midweek')
plt.plot(lin_pred_fri,color=o3,label='friday')
plt.plot(lin_pred_sat,color=o5,label='saturday')
plt.plot(lin_pred_sun,color=o7,label='sunday')
plt.plot(lin_pred_one_day,color='k',label='lin predicted all days')

plt.plot(loc_501_daily_mean,color=b3,label='mean')
plt.plot(loc_501_daily_median,color=b7,label='median')
plt.title('Hourly number of observations')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)


# %% OPTION 3
lin_reg_alld,lin_rmse_alld,lin_rmse_alld_cross,lin_501_predict_alld=linear_model(X_501_train_alld,y_501_train)

# Plot the Seven Options 
X_one_day_mon=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_one,array_zero,array_zero,array_zero,array_zero,array_zero,array_zero]))
X_one_day_tue=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_one,array_zero,array_zero,array_zero,array_zero,array_zero]))
X_one_day_wed=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_one,array_zero,array_zero,array_zero,array_zero]))
X_one_day_thu=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_zero,array_one,array_zero,array_zero,array_zero]))
X_one_day_fri=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_zero,array_zero,array_one,array_zero,array_zero]))
X_one_day_sat=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_zero,array_zero,array_zero,array_one,array_zero]))
X_one_day_sun=np.transpose(np.array([X_501_train_time['sin_time'][0:24],X_501_train_time['cos_time'][0:24],array_zero,array_zero,array_zero,array_zero,array_zero,array_zero,array_one]))

lin_pred_mon=lin_reg_alld.predict(X_one_day_mon)
lin_pred_tue=lin_reg_alld.predict(X_one_day_tue)
lin_pred_wed=lin_reg_alld.predict(X_one_day_wed)
lin_pred_thu=lin_reg_alld.predict(X_one_day_thu)
lin_pred_fri=lin_reg_alld.predict(X_one_day_fri)
lin_pred_sat=lin_reg_alld.predict(X_one_day_sat)
lin_pred_sun=lin_reg_alld.predict(X_one_day_sun)

plt.figure()
plt.plot(lin_pred_mon,color=o1,label='monday')
plt.plot(lin_pred_tue,color=o3,label='tuesday')
plt.plot(lin_pred_wed,color='c',label='wednesday')
plt.plot(lin_pred_thu,color=o7,label='thursday')
plt.plot(lin_pred_fri,color=b1,label='friday')
plt.plot(lin_pred_sat,color=b5,label='saturday')
plt.plot(lin_pred_sun,color=b9,label='sunday')
plt.plot(lin_pred_one_day,color='k',label='lin predicted all days')

plt.plot(loc_501_daily_mean,color=b3,label='mean')
plt.plot(loc_501_daily_median,color=b7,label='median')
plt.title('Hourly number of observations')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)


#%% Performance of all linear regression models
print('linear model RMSE:')
print(10*'*')
print('no dow:',lin_rmse_cross.mean())
print('week and weekend:',lin_rmse_wd_cross.mean())
print('ma-do,vr,za,zo:',lin_rmse_midw_cross.mean())
print('all dow:',lin_rmse_alld_cross.mean())
print(10*'*')

#%%
y_total=y_501_train.copy(deep=True)

y_pred_all=lin_reg_alld.predict(X_501_train_alld)
y_total['lin_pred_time']=y_pred_all
rmse_lin_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['lin_pred_time'])**2)/y_total.shape[0])
mae_lin_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['lin_pred_time']))/y_total.shape[0]

print('linear rmse time and dow:',rmse_lin_time)
print('linear mae time and dow',mae_lin_time)
# %% MODEL 2 DECISION TREE
# hyperparameter selection 
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,20],'min_samples_leaf' : [1,2,3,4]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_train_days_opt1,y_501_train)
print(grid_search.best_params_) #{'max_depth': 10, 'min_samples_leaf': 1} so best empty better if not constrained, maybe later for overfitting, holds for all 3 options
grid_search.best_estimator_ '''

# TODO HYPERPARAMETER OPTIMIZATION 
#OPTION 1
tree_wd,tree_rmse_wd,tree_rmse_wd_cross,tree_501_predict_wd=decision_tree_model(X_train_days_opt1,y_501_train,10,1)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_wd,400)


# %%OPTION 2

# Hyperparameter optimization
# hyperparameters: max_depth, min_samples_split,min_samples_leaf
# by GridsearchCv
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,14,13,12,15,16,17,20],'min_samples_leaf' : [1,2,3,4]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_day2,y_501_train)
print(grid_search.best_params_) #{'max_depth': 14, 'min_samples_leaf': 1}
grid_search.best_estimator_'''

tree_midw,tree_rmse_midw,tree_rmse_midw_cross,tree_501_predict_midw=decision_tree_model(X_501_train_day2,y_501_train,14,1)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_midw,400)

# %%OPTION 3 
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,14,13,12,15,16,17,20],'min_samples_leaf' : [1,2,3,4]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_alld,y_501_train)
print(grid_search.best_params_) #{'max_depth': 10, 'min_samples_leaf': 1}
grid_search.best_estimator_'''

tree_alld,tree_rmse_alld,tree_rmse_alld_cross,tree_501_predict_alld=decision_tree_model(X_501_train_alld,y_501_train,10,1)

y_pred_tree=tree_alld.predict(X_501_train_alld)
y_total['tree_pred_time']=y_pred_tree
rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

print('tree rmse time and dow:',rmse_tree_time)
print('tree mae time and dow',mae_tree_time)
# %% Plot what happens
#OPTION 1
tree_pred_week=tree_wd.predict(X_one_day_week)
tree_pred_weekend=tree_wd.predict(X_one_day_weekend)

plt.figure()
plt.plot(tree_pred_week,color=o1,label='tree predicted week')
plt.plot(tree_pred_weekend,color=o3,label='tree predicted weekend')
plt.plot(tree_reg_pred_one_day,color='k',label='tree predicted all days')
plt.plot(loc_501_daily_mean,color=b3,label='mean')
plt.plot(loc_501_daily_median,color=b7,label='median')
plt.title('Hourly number of observations')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)

# OPTION 2
tree_pred_midw=tree_midw.predict(X_one_day_midw)
tree_pred_fri=tree_midw.predict(X_one_day_fri_midw)
tree_pred_sat=tree_midw.predict(X_one_day_sat_midw)
tree_pred_sun=tree_midw.predict(X_one_day_sun_midw)

plt.figure()
plt.plot(tree_pred_midw,color=o1,label='midweek')
plt.plot(tree_pred_fri,color=o3,label='friday')
plt.plot(tree_pred_sat,color=o5,label='saturday')
plt.plot(tree_pred_sun,color=o7,label='sunday')
plt.plot(tree_reg_pred_one_day,color='k',label='tree predicted all days')

plt.plot(loc_501_daily_mean,color=b3,label='mean')
plt.plot(loc_501_daily_median,color=b7,label='median')
plt.title('Hourly number of observations')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)

#OPTION 3
tree_pred_mon=tree_alld.predict(X_one_day_mon)
tree_pred_tue=tree_alld.predict(X_one_day_tue)
tree_pred_wed=tree_alld.predict(X_one_day_wed)
tree_pred_thu=tree_alld.predict(X_one_day_thu)
tree_pred_fri=tree_alld.predict(X_one_day_fri)
tree_pred_sat=tree_alld.predict(X_one_day_sat)
tree_pred_sun=tree_alld.predict(X_one_day_sun)

plt.figure()
plt.plot(tree_pred_mon,color=o1,label='monday')
plt.plot(tree_pred_tue,color=o3,label='tuesday')
plt.plot(tree_pred_wed,color='c',label='wednesday')
plt.plot(tree_pred_thu,color=o7,label='thursday')
plt.plot(tree_pred_fri,color=b1,label='friday')
plt.plot(tree_pred_sat,color=b5,label='sunday')
plt.plot(tree_reg_pred_one_day,color=b9,label='tree predicted all days')

plt.plot(loc_501_daily_mean,label='mean')
plt.plot(loc_501_daily_median,label='median')
plt.title('Hourly number of observations')
plt.xlabel('Time')
plt.ylabel('Intensity')
plt.legend()
plt.xticks(rotation=80)


# %%print all performance measures decision tree
print('Decision tree RMSE:')
print(10*'*')
print('no dow:',tree_rmse_cross.mean()) 
print('week and weekend:',tree_rmse_wd_cross.mean()) 
print('ma-do,vr,za,zo:',tree_rmse_midw_cross.mean())
print('all dow:',tree_rmse_alld_cross.mean())
print(10*'*')

# %% RANDOM FOREST 

# OPTION 1
'''param_grid=[{'max_depth':[9,10,11],'min_samples_leaf' : [23,24,25,26,27]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_train_days_opt1,y_501_train_nuldim)
print(grid_search.best_params_ ) # max depth: 10 en min_smaples_leaf=26
grid_search.best_estimator_'''

forest_reg_opt1,forest_rmse_opt1,forest_rmse_cross_opt1,forest_501_predict_opt1=random_forest_model(X_train_days_opt1,y_501_train,100,1,10,26)

# OPTION 2
'''param_grid=[{'max_depth':[5,6,7,8,9],'min_samples_leaf' : [15,16,17,18]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_train_days_opt1,y_501_train_nuldim)
print(grid_search.best_params_ ) # max depth: 8 en min_smaples_leaf=17
grid_search.best_estimator_'''

forest_reg_midw,forest_rmse_midw,forest_rmse_cross_midw,forest_501_predict_midw=random_forest_model(X_501_train_day2,y_501_train,100,1,8,17)

# OPTION 3
'''param_grid=[{'max_depth':[8,9,10,11],'min_samples_leaf' : [14,15,16]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_alld,y_501_train_nuldim)
print(grid_search.best_params_ ) # max depth: 9 en min_smaples_leaf=15
grid_search.best_estimator_'''

forest_reg_alld,forest_rmse_alld,forest_rmse_cross_alld,forest_501_predict_alld=random_forest_model(X_501_train_alld,y_501_train,100,1,9,15)

y_pred_tree=forest_reg_alld.predict(X_501_train_alld)
y_total['tree_pred_time']=y_pred_tree
rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

print('forest rmse time and dow:',rmse_tree_time)
print('forest mae time and dow',mae_tree_time)

# %%print all performance measures random forest
print('Random forest RMSE:')
print(10*'*')
print('no dow:',forest_rmse_cross.mean()) 
print('week and weekend:',forest_rmse_cross_opt1.mean()) 
print('ma-do,vr,za,zo:',forest_rmse_cross_midw.mean())
print('all dow:',forest_rmse_cross_alld.mean())
print(10*'*')

# %% SUPPORT VECTOR MACHINE LINEAR
# LINEAR
# OPTION 1
svm_reg_opt1,svm_rmse_opt1,svm_501_predict_opt1=svm_linear(X_train_days_opt1,y_501_train,1)

# OPTION 2
svm_reg_midw,svm_rmse_midw,svm_501_predict_midw=svm_linear(X_501_train_day2,y_501_train,1)

#OPTION 3
svm_reg_alld,svm_rmse_alld,svm_501_predict_alld=svm_linear(X_501_train_alld,y_501_train,1)

# %% NON LINEAR
# OPTION 1


svm_reg_nl_opt1,svm_rmse_nl_opt1,svm_501_predict_nl_opt1=svm_nonlinear(X_train_days_opt1,y_501_train,'poly',10,1)

#%% OPTION 2
svm_reg_nl_midw,svm_rmse_nl_midw,svm_501_predict_nl_midw=svm_nonlinear(X_501_train_day2,y_501_train,'poly',3,1)

# OPTION 3
svm_reg_nl_alld,svm_rmse_nl_alld,svm_501_predict_nl_alld=svm_nonlinear(X_501_train_alld,y_501_train,'poly',3,1)


# %% print all performance measures support vector machine
print('Linear SVM RMSE:')
print(10*'*')
print('no dow:',svm_rmse) 
print('week and weekend:',svm_rmse_opt1) 
print('ma-do,vr,za,zo:',svm_rmse_midw)
print('all dow:',svm_rmse_alld)
print(10*'*')

print('Non linear SVM RMSE:')
print(10*'*')
print('no dow:',svm_rmse_nl) 
print('week and weekend:',svm_rmse_nl_opt1) 
print('ma-do,vr,za,zo:',svm_rmse_nl_midw)
print('all dow:',svm_rmse_nl_alld)
print(10*'*')

#%%  NEXT STEP, ADD SEASON FEATURE
# two options:
#       A. winter or summer
#       B. sinus, so continuous variable instead of discrete


# OPTION 1. SUMMER AND WINTER

# linear
lin_reg_season1,lin_rmse_season1,lin_rmse_season1_cross,lin_501_predict_season1=linear_model(X_501_train_season,y_501_train)
plot_errors_on_day(loc_501_train, y_501_train, lin_501_predict_season1,400)

print(lin_rmse_season1)


y_pred_tree=lin_reg_season1.predict(X_501_train_season)
y_total['tree_pred_time']=y_pred_tree
rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

print('lin rmse time and dow season:',rmse_tree_time)
print('lin mae time and dow season',mae_tree_time)


# decision tree
tree_season1,tree_rmse_season1,tree_rmse_cross_season1,tree_501_predict_season1=decision_tree_model(X_501_train_season,y_501_train,[],[])
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_season1,400)


# random forest
forest_reg_season1,forest_rmse_season1,forest_rmse_cross_season1,forest_501_predict_season1=random_forest_model(X_501_train_season,y_501_train,100,1,[],[])
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_season1,400)

# linear svm
svm_reg_season1,svm_rmse_season1,svm_501_predict_season1=svm_linear(X_501_train_season,y_501_train,1)
plot_errors_on_day(loc_501_train, y_501_train, svm_501_predict_season1,400)

# non linear svm
svm_reg_nl_season1,svm_rmse_nl_season1,svm_501_predict_nl_season1=svm_nonlinear(X_501_train_season,y_501_train,'poly',3,1)
plot_errors_on_day(loc_501_train, y_501_train, svm_501_predict_nl_season1,400)

# Print results
print('RMSE summer winter [0 or 1]')
print(10*'*')
print('linear:', lin_rmse_season1)
print('decision tree:', tree_rmse_season1)
print('decision tree cross:', tree_rmse_cross_season1.mean())
print('random forest:',forest_rmse_season1)
print('svm linear:',svm_rmse_season1)
print('svm nonlinear:',svm_rmse_nl_season1)

# %% Optimize DECISION TREE

# hyperparameters: max_depth,min_samples_leaf
# by GridsearchCv does cross validation!

'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'min_samples_leaf' : [1,2,3,4,5,10,15,20,25,30,35]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_season,y_501_train)
print(grid_search.best_params_ )#13 en 1
grid_search.best_estimator_'''

# final model
tree_reg,tree_rmse,tree_rmse_cross,tree_501_predict=decision_tree_model(X_501_train_season,y_501_train,13,1)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict,400)

# %% Optimize RANDOM FOREST
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'min_samples_leaf' : [1,2,3,4,5,10,15,20,25,30,35]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_season,y_501_train_nuldim)
print(grid_search.best_params_ ) # max depth: 12 en min_smaples_leaf=4
grid_search.best_estimator_'''

# final model
forest_reg_season1,forest_rmse_season1,forest_rmse_cross_season1,forest_501_predict_season1=random_forest_model(X_501_train_season,y_501_train,100,1,12,4)
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_season1,400)

y_pred_tree=forest_reg_season1.predict(X_501_train_season)
y_total['tree_pred_time']=y_pred_tree
rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

print('forest rmse time and dow season:',rmse_tree_time)
print('forest mae time and dow season',mae_tree_time)
# %%OPTION 2. COSINUS
X_501_train_season_opt2=X_501_train_day2.copy(deep=True)

days_in_year=np.linspace(0,364,365)
cos_time_season=np.cos(2*np.pi*days_in_year/365) 
n_years=3
cos_time_season_three_years=[]
for i in range(n_years):
    cos_time_season_three_years=np.append(cos_time_season_three_years,cos_time_season)

all_days_array=np.repeat(cos_time_season_three_years,24) #because all 24 hours in the day have the same value
all_day_df=pd.DataFrame(data={'all_days':all_days_array})

season_nodig=all_day_df.loc[original_index_all['original_index'],:].reset_index()
X_501_train_season_opt2['season']=season_nodig['all_days']          

#%%
# linear
lin_reg_season2,lin_rmse_season2,lin_rmse_season2_cross,lin_501_predict_season2=linear_model(X_501_train_season_opt2,y_501_train)
plot_errors_on_day(loc_501_train, y_501_train, lin_501_predict_season2,400)

# decision tree
tree_season2,tree_rmse_season2,tree_rmse_cross_season2,tree_501_predict_season2=decision_tree_model(X_501_train_season_opt2,y_501_train,[],[])
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_season2,400)

# random forest
forest_reg_season2,forest_rmse_season2,forest_rmse_cross_season2,forest_501_predict_season2=random_forest_model(X_501_train_season_opt2,y_501_train,100,1,[],[])
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_season2,100)

# linear svm
svm_reg_season2,svm_rmse_season2,svm_501_predict_season2=svm_linear(X_501_train_season_opt2,y_501_train,1)
plot_errors_on_day(loc_501_train, y_501_train, svm_501_predict_season2,400)

# non linear svm
svm_reg_nl_season2,svm_rmse_nl_season2,svm_501_predict_nl_season2=svm_nonlinear(X_501_train_season_opt2,y_501_train,'poly',3,1)
plot_errors_on_day(loc_501_train, y_501_train, svm_501_predict_nl_season2,400)


# %% je ziet aan de random forest en decision tree gekke dingen, misschien wordt die geoverfit, decision tree normal en cross groot verschil en random forest ineens super laag

# Hyperparameter optimization DECISION TREE
# hyperparameters: max_depth, min_samples_split,min_samples_leaf
# by GridsearchCv does cross validation!
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],'min_samples_leaf' : [1,2,3,4,5,10,15,20,25,26,27,28,29,30,31,32,33,34,35]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_season_opt2,y_501_train)
print(grid_search.best_params_ )#7 en 33
grid_search.best_estimator_'''
# final model
tree_reg_season2,tree_rmse_season2,tree_rmse_cross_season2,tree_501_predict_season2=decision_tree_model(X_501_train_season_opt2,y_501_train,7,33)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_season2,400)

# %% Optimize RANDOM FOREST
'''param_grid=[{'max_depth':[8,9,10,11,12,13,15],'min_samples_leaf' : [17,18,19,20,21,22,23]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_season_opt2,y_501_train_nuldim)
print(grid_search.best_params_ ) # 13 en 19
grid_search.best_estimator_'''

# final model
forest_reg_season2,forest_rmse_season2,forest_rmse_cross_season2,forest_501_predict_season2=random_forest_model(X_501_train_season_opt2,y_501_train,100,1,13,19)
#plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_season2,400)

# %% Print results summer winter option 2
print('RMSE summer winter cosinus')
print(10*'*')
print('linear:', lin_rmse_season2)
print('decision tree:', tree_rmse_season2)
print('decision tree cross:', tree_rmse_cross_season2.mean())
print('random forest:',forest_rmse_season2)
print('random forest cross:',forest_rmse_cross_season2.mean())
print('svm linear:',svm_rmse_season2)
print('svm nonlinear:',svm_rmse_nl_season2)


#%% Next step, FEESTDAGEN 

# 2 opties, 
    # 1: feestdagen en normale dagen allebei een aparte feature
    # 2: alleen een feature voor feestdagen --> logischer
X_501_train_feestdag['start_datum']=loc_501_train['start_datum']
X_501_train_feestdag_calplot=X_501_train_feestdag.copy(deep=True)
X_501_train_feestdag_calplot=X_501_train_feestdag.set_index('start_datum')

calplot.calplot(X_501_train_feestdag_calplot['feestdag'],cmap=cmap_blue_orange_2,textfiller='-',colorbar=False) #hier nice kleur map toevoegen maar ligt aan hoeveel clusters
plot_clusters(X_501_clus,5,cmap_blue_orange_6,colors_6)

# TODO: eigenlijk een nice plotje van de evenementen welke kleur cluster dit normaal in mijn algemene is, een gekke of een normaal cluster
# drop start datum
X_501_train_feestdag=X_501_train_feestdag.drop('start_datum',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# optie 2
#X_501_train_holiday2=X_501_train_holiday.drop(['normal_day'],axis=1)
# %% Fit all models

# linear
lin_reg_holiday,lin_rmse_holiday,lin_rmse_cross_holiday,lin_501_predict_holiday=linear_model(X_501_train_feestdag,y_501_train)
#lin_reg_holiday2,lin_rmse_holiday2,lin_rmse_cross_holiday2,lin_501_predict_holiday2=linear_model(X_501_train_holiday2,y_501_train)

plot_errors_on_day(loc_501_train, y_501_train, lin_501_predict_holiday,400)

y_pred_tree=lin_reg_holiday.predict(X_501_train_feestdag)
y_total['tree_pred_time']=y_pred_tree
rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

print('lin rmse time and dow season holiday:',rmse_tree_time)
print('lin mae time and dow season holiday',mae_tree_time)


# decision tree
tree_holiday,tree_rmse_holiday,tree_rmse_cross_holiday,tree_501_predict_holiday=decision_tree_model(X_501_train_feestdag,y_501_train,[],[])
#tree_holiday2,tree_rmse_holiday2,tree_rmse_cross_holiday2,tree_501_predict_holiday2=decision_tree_model(X_501_train_holiday2,y_501_train,[],[])

plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_holiday,400)

#%% hyperparameter optimization decision tree
param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13],'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_feestdag,y_501_train)
print(grid_search.best_params_ ) # 13 en 9
grid_search.best_estimator_

# decision tree optimized
tree_holiday,tree_rmse_holiday,tree_rmse_cross_holiday,tree_501_predict_holiday=decision_tree_model(X_501_train_feestdag,y_501_train,13,9)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_holiday,400)
'''
param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13],'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday2,y_501_train)
print(grid_search.best_params_ ) # 10 en 5
grid_search.best_estimator_

# decision tree optimized
tree_holiday2,tree_rmse_holiday2,tree_rmse_cross_holiday2,tree_501_predict_holiday2=decision_tree_model(X_501_train_holiday2,y_501_train,10,5)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_holiday2,400)
'''

# %%random forest
forest_reg_holiday,forest_rmse_holiday,forest_rmse_cross_holiday,forest_501_predict_holiday=random_forest_model(X_501_train_feestdag,y_501_train,100,1,[],[])
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday,400)

#forest_reg_holiday2,forest_rmse_holiday2,forest_rmse_cross_holiday2,forest_501_predict_holiday2=random_forest_model(X_501_train_holiday2,y_501_train,100,1,[],[])


# %% hyperparameter optimization random forest 
'''param_grid=[{'max_depth':[4,6,8,9,10,11,12,13,15,20],'min_samples_leaf' : [3,4,5,6,7,10,17,18]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday,y_501_train_nuldim)
print(grid_search.best_params_ ) # 15 en 4
grid_search.best_estimator_'''

# random forest
forest_reg_holiday,forest_rmse_holiday,forest_rmse_cross_holiday,forest_501_predict_holiday=random_forest_model(X_501_train_feestdag,y_501_train,100,1,15,4)
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday,400)
'''
param_grid=[{'max_depth':[4,6,8,9,10,11,12,13,15,20],'min_samples_leaf' : [3,4,5,6,7,10,17,18]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday2,y_501_train_nuldim)
print(grid_search.best_params_ ) # 12 en 5
grid_search.best_estimator_
'''
# random forest
#forest_reg_holiday2,forest_rmse_holiday2,forest_rmse_cross_holiday2,forest_501_predict_holiday2=random_forest_model(X_501_train_holiday2,y_501_train,100,1,12,5)
#plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday2,400)


#y_pred_tree=forest_reg_holiday2.predict(X_501_train_holiday2)
#y_total['tree_pred_time']=y_pred_tree
#rmse_tree_time=np.sqrt(sum((y_total['waarnemingen_intensiteit']-y_total['tree_pred_time'])**2)/y_total.shape[0])
#mae_tree_time=sum(abs(y_total['waarnemingen_intensiteit']-y_total['tree_pred_time']))/y_total.shape[0]

#print('forest rmse time and dow season holiday:',rmse_tree_time)
#print('forest mae time and dow season holiday',mae_tree_time)

# %%linear svm
svm_reg_holiday,svm_rmse_holiday,svm_501_predict_holiday=svm_linear(X_501_train_feestdag,y_501_train,1)
plot_errors_on_day(loc_501_train, y_501_train, svm_501_predict_holiday,400)

# non linear svm
svm_reg_nl_holiday,svm_rmse_nl_holiday,svm_501_predict_nl_holiday=svm_nonlinear(X_501_train_feestdag,y_501_train,'poly',3,1)
plot_errors_on_day(loc_501_train, y_501_train, svm_501_predict_nl_holiday,400)

# Print results
print('RMSE summer winter cosinus')
print(10*'*')
print('linear:', lin_rmse_holiday)
print('decision tree:', tree_rmse_holiday)
print('decision tree cross:', tree_rmse_cross_holiday.mean())
print('random forest:',forest_rmse_holiday)
print('svm linear:',svm_rmse_holiday)
print('svm nonlinear:',svm_rmse_nl_holiday)

# %% check de invloed van holiday toevoegen op een feestdag voor Random forest

#look at data 2019-01-01','2019-04-19','2019-04-21','2019-04-22','2019-04-27','2019-05-05','2019-05-30','2019-06-09','2019-06-10'
feestdagen_2019=['2019-01-01','2019-04-19','2019-04-21','2019-04-22','2019-04-27','2019-05-05','2019-05-30','2019-06-09','2019-06-10','2019-12-25','2019-12-26','2019-12-31']
feestdagen_2018=['2018-01-01','2018-03-30','2018-04-01','2018-04-02','2018-04-27','2018-05-05','2018-05-10','2018-05-20','2018-05-21','2018-12-25','2018-12-26','2018-12-31']
feestdagen_2017=['2017-01-01','2017-04-14','2017-04-16','2017-04-17','2017-04-27','2017-05-05','2017-05-25','2017-06-04','2017-06-05','2017-12-25','2017-12-26','2017-12-31']
# nieuwjaarsdag, goede vrijdag, pasen, bevrijdingsdag,hemelvaartsdag, pinksteren,kerst, Zijn maar 12 dagen, dus 36 in totaal
# 2017-05-05 is verwijderd uit de data en 2018-12-31

data_to_look_at=feestdagen_2019+feestdagen_2018+feestdagen_2017

#%%
for i in range(len(data_to_look_at)): 
    print('i',i)
    date=data_to_look_at[i]
    index_1=np.where(loc_501_train['start_datum']==date)
    y_1=y_501_train.loc[X_501_train_time.index[index_1],:]
    y_1=y_1.reset_index(inplace=False)
    y_1=y_1.drop('index',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

    # prediction normal
    X_data_1=X_501_train_time.loc[X_501_train_time.index[index_1],:]
    forest_normal_1=forest_reg.predict(X_data_1)
       
    # prediction dow
    X_data_1=X_501_train_dow.loc[X_501_train_time.index[index_1],:]
    forest_dow_1=forest_reg_alld.predict(X_data_1)
    
    # prediction season
    X_data_1=X_501_train_season.loc[X_501_train_time.index[index_1],:]
    forest_season_1=forest_reg_season1.predict(X_data_1)
    
    # prediction holiday
    X_data_1=X_501_train_feestdag.loc[X_501_train_time.index[index_1],:]
    forest_holiday_1=forest_reg_holiday.predict(X_data_1)
    
    
    plt.figure()
    plt.plot(forest_normal_1,label='time ')
    plt.plot(forest_dow_1,label='dow')
    plt.plot(forest_season_1,label='season')
    plt.plot(forest_holiday_1,label='holiday')
    plt.plot(y_1,label='true')
    plt.legend()
    plt.title(data_to_look_at[i])


#%% Next step, vacation period
# TOT HIEEEEEER gekeken met nieuwe features van 3 jaar.

#Noord:  voorjaarsvakantie: 16/02-24/02
#(Haarlem) meivakantie: 27/04-05/05
        # zomer 13/07-25/08
        #  herfst 19/10-27/10
        # kerst 21/12 05/01/2020
        # kerst tweedeze tm 06/01
        
#Midden:  voorjaarsvakantie: 23/02-03/03
        # meivakantie 27/04 -05/05
        # zomer 20/07-01/09
        # herfst 19/10 - 27/10
        # kerst 21/12-05/01/2020
        
#Zuid: voorjaars 23/02 - 03/03
        # meivakantie: 27/04 -05/05
        # zomervakantie: 06/07-18/08
        # herfst: 12/10-20/10
        # kerst 21/12-05/01/2020

# alleen voorjaars, zomer,herfst zijn anders bij deze, 3 opties:
    # 1. noord midden zuid apart meenemen wanneer vakantie
    # 2. noord midden zuid als een feature meenemen wanneer vakantie
    # 3. een van de bovenste twee in combinatie met de officiele feestdagen

# TO Do maak plaatje van al deze dagen die hierbij behoren van calplot met de kleur van bijbehorend cluster. 
    # 1. van 4 clusters
    # 2. van 5 clusters

# Voor vier clusters
X_total=X_501_4.copy(deep=True)

X_feestdagen=X_total.loc[X_total.index.isin(data_to_look_at)]
X_voorjaar=X_total.loc[(X_total.index>='2019-02-16') &(X_total.index<='2019-02-24'),:]
X_zomer=X_total.loc[(X_total.index>='2019-07-13') &(X_total.index<='2019-08-25'),:]
X_herfst=X_total.loc[(X_total.index>='2019-10-19') &(X_total.index<='2019-10-27'),:]
X_kerst=X_total.loc[(X_total.index>='2019-12-21') &(X_total.index<='2019-12-31'),:]
X_kerst2=X_total.loc[X_total.index<='2019-01-06',:]

X_voorjaar_ander=X_total.loc[(X_total.index>='2019-02-23') &(X_total.index<='2019-03-03'),:]
X_zomer_ander=X_total.loc[(X_total.index>'2019-08-25') &(X_total.index<='2019-09-01'),:]
X_zomer_ander2=X_total.loc[(X_total.index>='2019-07-06') &(X_total.index<'2019-07-13'),:]
X_herfst_ander=X_total.loc[(X_total.index>='2019-10-12') &(X_total.index<='2019-10-20'),:]

X_holiday=pd.concat([X_feestdagen,X_voorjaar,X_zomer,X_herfst,X_kerst,X_kerst2,X_voorjaar_ander,X_zomer_ander,X_zomer_ander2,X_herfst_ander])
cmap_try=LinearSegmentedColormap.from_list("",[o1,o7,b1,b7])

calplot.calplot(X_holiday['label'],cmap=cmap_try,textfiller='-',colorbar=True) #hier nice kleur map toevoegen maar ligt aan hoeveel clusters

# Voor 5 clusters
X_total=X_501_5.copy(deep=True)

# Alle vakantie en feestdagen
X_feestdagen=X_total.loc[X_total.index.isin(data_to_look_at)]
X_voorjaar=X_total.loc[(X_total.index>='2019-02-16') &(X_total.index<='2019-02-24'),:]
X_zomer=X_total.loc[(X_total.index>='2019-07-13') &(X_total.index<='2019-08-25'),:]
X_herfst=X_total.loc[(X_total.index>='2019-10-19') &(X_total.index<='2019-10-27'),:]
X_kerst=X_total.loc[(X_total.index>='2019-12-21') &(X_total.index<='2019-12-31'),:]
X_kerst2=X_total.loc[X_total.index<='2019-01-06',:]

X_voorjaar_ander=X_total.loc[(X_total.index>='2019-02-23') &(X_total.index<='2019-03-03'),:]
X_zomer_ander=X_total.loc[(X_total.index>'2019-08-25') &(X_total.index<='2019-09-01'),:]
X_zomer_ander2=X_total.loc[(X_total.index>='2019-07-06') &(X_total.index<'2019-07-13'),:]
X_herfst_ander=X_total.loc[(X_total.index>='2019-10-12') &(X_total.index<='2019-10-20'),:]

X_holiday=pd.concat([X_feestdagen,X_voorjaar,X_zomer,X_herfst,X_kerst,X_kerst2,X_voorjaar_ander,X_zomer_ander,X_zomer_ander2,X_herfst_ander])
cmap_try=LinearSegmentedColormap.from_list("",[b1,b7,o1,o4,o7])

calplot.calplot(X_holiday['label'],cmap=cmap_try,textfiller='-',colorbar=True) #hier nice kleur map toevoegen maar ligt aan hoeveel clusters



# %%Stap 1, maak feature space met alles bij elkaar als 1 feature en normale dagen als de andere feature
X_501_train_holiday_full=X_501_train_season.copy(deep=True)
X_501_train_holiday_full['start_datum']=loc_501_train['start_datum']
X_501_train_holiday_full['holiday']=0
X_501_train_holiday_full['normal_day']=0

all_holiday_dates=X_holiday.index
condition_holiday=X_501_train_holiday_full['start_datum'].isin(all_holiday_dates)
X_501_train_holiday_full.loc[X_501_train_holiday_full.index[condition_holiday],'holiday']=1

Condition_normal=X_501_train_holiday_full['holiday']==0
X_501_train_holiday_full.loc[X_501_train_holiday_full.index[Condition_normal],'normal_day']=1

X_501_train_holiday_full=X_501_train_holiday_full.drop('start_datum',axis=1,inplace=False) #inplace is False zodat hij een copy geeft
X_501_train_holiday_full2=X_501_train_holiday_full.drop('normal_day',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# %% Fit models
# linear
lin_reg_holiday_full,lin_rmse_holiday_full,lin_rmse_cross_holiday_full,lin_501_predict_holiday_full=linear_model(X_501_train_holiday_full,y_501_train)
lin_reg_holiday_full2,lin_rmse_holiday_full2,lin_rmse_cross_holiday_full2,lin_501_predict_holiday_full2=linear_model(X_501_train_holiday_full2,y_501_train)
print('lin RMSE 1:',lin_rmse_cross_holiday_full.mean())
print('lin RMSE 2:',lin_rmse_cross_holiday_full2.mean())

plot_errors_on_day(loc_501_train, y_501_train, lin_501_predict_holiday_full,400)

# decision tree
tree_holiday_full,tree_rmse_holiday_full,tree_rmse_cross_holiday_full,tree_501_predict_holiday_full=decision_tree_model(X_501_train_holiday_full,y_501_train,[],[])
tree_holiday_full2,tree_rmse_holiday_full2,tree_rmse_cross_holiday_full2,tree_501_predict_holiday_full2=decision_tree_model(X_501_train_holiday_full2,y_501_train,[],[])
print('dt cross RMSE 1:',tree_rmse_cross_holiday_full.mean())
print('dt cross RMSE 2:',tree_rmse_cross_holiday_full2.mean())

plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_holiday_full,400)

#%% hyperparameter optimization decision tree
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13],'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday_full,y_501_train)
print(grid_search.best_params_ ) # 6 en 1 
grid_search.best_estimator_'''

# decision tree optimized
tree_holiday_full,tree_rmse_holiday_full,tree_rmse_cross_holiday_full,tree_501_predict_holiday_full=decision_tree_model(X_501_train_holiday_full,y_501_train,6,1)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_holiday_full,400)

'''
param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13],'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday_full2,y_501_train)
print(grid_search.best_params_ ) #6 en 1 '''

# decision tree optimized
tree_holiday_full2,tree_rmse_holiday_full2,tree_rmse_cross_holiday_full2,tree_501_predict_holiday_full2=decision_tree_model(X_501_train_holiday_full2,y_501_train,6,1)
plot_errors_on_day(loc_501_train, y_501_train, tree_501_predict_holiday_full2,400)


# %%random forest
forest_reg_holiday_full,forest_rmse_holiday_full,forest_rmse_cross_holiday_full,forest_501_predict_holiday_full=random_forest_model(X_501_train_holiday_full,y_501_train,100,1,[],[])
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday,400)

forest_reg_holiday_full2,forest_rmse_holiday_full2,forest_rmse_cross_holiday_full2,forest_501_predict_holiday_full2=random_forest_model(X_501_train_holiday_full2,y_501_train,100,1,[],[])


# %% hyperparameter optimization random forest 
'''param_grid=[{'max_depth':[11,12,13,14,15],'min_samples_leaf' : [15,16,17,18]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday_full,y_501_train_nuldim)
print(grid_search.best_params_ ) # 12 en 17
grid_search.best_estimator_'''

# random forest
forest_reg_holiday_full,forest_rmse_holiday_full,forest_rmse_cross_holiday_full,forest_501_predict_holiday_full=random_forest_model(X_501_train_holiday_full,y_501_train,100,1,12,17)
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday,400)

'''param_grid=[{'max_depth':[12,13,14],'min_samples_leaf' : [15,16,17,18]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday_full2,y_501_train_nuldim)
print(grid_search.best_params_ ) # 13 en 16
grid_search.best_estimator_'''

# random forest
forest_reg_holiday_full2,forest_rmse_holiday_full2,forest_rmse_cross_holiday_full2,forest_501_predict_holiday_full2=random_forest_model(X_501_train_holiday_full2,y_501_train,100,1,13,16)
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday_full2,400)

# %% Probeer vakanties en feestdagen en zomervakantie apart


X_holiday_opt2=pd.concat([X_feestdagen,X_voorjaar,X_herfst,X_kerst,X_kerst2,X_voorjaar_ander,X_herfst_ander])
X_zomervak_opt2=pd.concat([X_zomer,X_zomer_ander,X_zomer_ander2])

# Stap 1, maak feature space met alles bij elkaar als 1 feature en normale dagen als de andere feature
X_501_train_holiday_sep=X_501_train_season.copy(deep=True)
X_501_train_holiday_sep['start_datum']=loc_501_train['start_datum']
X_501_train_holiday_sep['summer_vacation']=0
X_501_train_holiday_sep['holiday']=0

holiday_dates=X_holiday_opt2.index
summer_dates=X_zomervak_opt2.index

condition_holiday=X_501_train_holiday_sep['start_datum'].isin(holiday_dates)
condition_summer=X_501_train_holiday_sep['start_datum'].isin(summer_dates)
X_501_train_holiday_sep.loc[X_501_train_holiday_sep.index[condition_holiday],'holiday']=1
X_501_train_holiday_sep.loc[X_501_train_holiday_sep.index[condition_summer],'summer_vacation']=1

X_501_train_holiday_sep=X_501_train_holiday_sep.drop('start_datum',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# %%decision tree
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13],'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday_sep,y_501_train)
print(grid_search.best_params_ ) #11 en 7 '''

# decision tree optimized
tree_holiday_full_sep,tree_rmse_holiday_full2,tree_rmse_cross_holiday_full2,tree_501_predict_holiday_full2=decision_tree_model(X_501_train_holiday_sep,y_501_train,11,7)
plot_errors_on_day(loc_501_train, y_501_train, X_501_train_holiday_sep,400)

# %%random forest
'''param_grid=[{'max_depth':[18,19,20,21],'min_samples_leaf' : [13,14,15]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_holiday_sep,y_501_train_nuldim)
print(grid_search.best_params_ ) # 21 en 14
grid_search.best_estimator_'''

# random forest
forest_reg_holiday_full2,forest_rmse_holiday_full2,forest_rmse_cross_holiday_full2,forest_501_predict_holiday_full2=random_forest_model(X_501_train_holiday_sep,y_501_train,100,1,21,14)
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday_full2,400)

# %% Optie : feestdagen en vakantie dagen apart
X_feestdagen_dates=X_feestdagen.index
X_vakantie=pd.concat([X_voorjaar,X_zomer,X_herfst,X_kerst,X_kerst2,X_voorjaar_ander,X_zomer_ander,X_zomer_ander2,X_herfst_ander])
X_vakantie_dates=X_vakantie.index

X_501_train_vak_en_feest=X_501_train_season.copy(deep=True)
X_501_train_vak_en_feest['start_datum']=loc_501_train['start_datum']
X_501_train_vak_en_feest['vakantie']=0
X_501_train_vak_en_feest['feestdag']=0


condition_feestdagen=X_501_train_vak_en_feest['start_datum'].isin(X_feestdagen_dates)
condition_vakantie=X_501_train_vak_en_feest['start_datum'].isin(X_vakantie_dates)
X_501_train_vak_en_feest.loc[X_501_train_vak_en_feest.index[condition_feestdagen],'feestdag']=1
X_501_train_vak_en_feest.loc[X_501_train_vak_en_feest.index[condition_vakantie],'vakantie']=1

X_501_train_vak_en_feest=X_501_train_vak_en_feest.drop('start_datum',axis=1,inplace=False) #inplace is False zodat hij een copy geeft

# %% Decision tree
'''param_grid=[{'max_depth':[2,3,4,5,6,7,8,9,10,11,12,13],'min_samples_leaf' : [1,2,3,4,5,6,7,8,9,10]}]
tree_reg_h=DecisionTreeRegressor()
grid_search=GridSearchCV(tree_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_vak_en_feest,y_501_train)
print(grid_search.best_params_ ) #12 en 9 '''

# decision tree optimized
tree_holiday_full_sep,tree_rmse_holiday_full2,tree_rmse_cross_holiday_full2,tree_501_predict_holiday_full2=decision_tree_model(X_501_train_vak_en_feest,y_501_train,12,9)
plot_errors_on_day(loc_501_train, y_501_train, X_501_train_holiday_sep,400)


# %% Random forest
'''param_grid=[{'max_depth':[10,11,12,15,18,19,20,24],'min_samples_leaf' : [1,5,8,13,14,15,18]}]
forest_reg_h=RandomForestRegressor()
grid_search=GridSearchCV(forest_reg_h, param_grid,cv=10,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_501_train_vak_en_feest,y_501_train_nuldim)
print(grid_search.best_params_ ) # 21 en 14
grid_search.best_estimator_'''

# random forest
forest_reg_holiday_full2,forest_rmse_holiday_full2,forest_rmse_cross_holiday_full2,forest_501_predict_holiday_full2=random_forest_model(X_501_train_vak_en_feest,y_501_train,100,1,11,10)
plot_errors_on_day(loc_501_train, y_501_train, forest_501_predict_holiday_full2,400)


# %% Next step weather


# %% Next step events




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



''' go to one hot encoding
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()'''
    # 2 ma-do, vr,za,zo
    # 3 ma,di,wo,do,vr,za,zo