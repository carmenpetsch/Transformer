# This file makes the Random forest

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
from baseline_models_functions import plot_errors_on_day
from baseline_models_functions import random_forest_model

# machine learning models
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import tree 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR




# machine learning performance metrics
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score #for cross validation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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
siemens_blauw_groen=(0/255,90/255,120/255)

colors_2=[b3,b7]
colors_3=[b3,b7,o3]
colors_4=[b3,b7,o3,o7]
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

#%% Make final data set
# NOTE CHANGED
original_index_501=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_501.pkl")


def import_data_location(location):
    loc=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_{}.pkl".format(location))
    X_train=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/X_{}_train_final.pkl".format(location))
    X_test=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/X_{}_test_final.pkl".format(location))
    y_train=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/y_{}_train.pkl".format(location))
    y_test=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/y_{}_test.pkl".format(location))
    y_train_stand_values=stats.zscore(y_train.values).reshape(y_train.shape[0],)
    y_train_norm=(y_train-y_train.min())/(y_train.max()-y_train.min()) #min max normalization
    y_train_stand=pd.DataFrame({'waarnemingen_intensiteit':y_train_stand_values})

    
    
    y_train_mean=np.mean(y_train.values)
    y_train_std=np.std(y_train.values)
    
    y_test_stand=(y_test-y_train_mean)/y_train_std

    
    return loc,X_train,X_test,y_train,y_test,y_train_stand,y_test_stand,y_train_norm,y_train_mean,y_train_std
[loc_501,X_501_train,X_501_test,y_501_train,y_501_test,y_501_train_stand,y_501_test_stand,y_501_train_norm,y_501_mean,y_501_std]=import_data_location('501')

X_501_train_time=X_501_train.copy(deep=True) # copy fine because now includes everything#pd.DataFrame({'sin_time':X_501_train['sin_time'],'cos_time':X_501_train['cos_time'],'mon':X_501_train['mon'], 'tue':X_501_train['tue'], 'wed':X_501_train['wed'], 'thu':X_501_train['thu'], 'fri':X_501_train['fri'], 'sat':X_501_train['sat'], 'sun':X_501_train['sun']})#,'season':X_501_train['season']})
X_501_test_time=X_501_test.copy(deep=True)

del X_501_train_time['Sun_duration']
del X_501_test_time['Sun_duration']

# Step 1 add extra season feature
days_in_year=np.linspace(0,364,365)
sin_time_season=np.sin(2*np.pi*days_in_year/365) 
n_years=3
sin_time_season_three_years=[]
for i in range(n_years):
    sin_time_season_three_years=np.append(sin_time_season_three_years,sin_time_season)

all_days_array=np.repeat(sin_time_season_three_years,24) #because all 24 hours in the day have the same value
all_day_df=pd.DataFrame(data={'all_days':all_days_array})

season_nodig=all_day_df.loc[original_index_501['original_index'],:].reset_index(drop=True)


# 1. split the set again in test and train

X_total=pd.concat([X_501_train_time,X_501_test_time])
X_total['season2']=season_nodig
y_total=pd.concat([y_501_train_stand,y_501_test_stand])

# Step 4 add school holiday as a feature
X_total['vacation']=0


#2017
X_total.loc[loc_501['start_datum']<='2017-01-08','vacation']=1
X_total.loc[(loc_501['start_datum']>='2017-02-18' )& (loc_501['start_datum']<='2017-02-26'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2017-04-22' )& (loc_501['start_datum']<='2017-05-05'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2017-07-22' )& (loc_501['start_datum']<='2017-09-03'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2017-10-21' )& (loc_501['start_datum']<='2017-10-29'),'vacation']=1
#2018
X_total.loc[(loc_501['start_datum']>='2017-12-23' )& (loc_501['start_datum']<='2018-01-07'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2018-02-24' )& (loc_501['start_datum']<='2018-03-04'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2018-04-28' )& (loc_501['start_datum']<='2018-05-13'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2018-07-21' )& (loc_501['start_datum']<='2018-09-02'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2018-10-20' )& (loc_501['start_datum']<='2018-10-28'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2018-12-22' )& (loc_501['start_datum']<='2019-01-06'),'vacation']=1
#2019
X_total.loc[(loc_501['start_datum']>='2019-02-15' )& (loc_501['start_datum']<='2019-02-24'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2019-04-20' )& (loc_501['start_datum']<='2019-05-05'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2019-07-13' )& (loc_501['start_datum']<='2019-08-25'),'vacation']=1
X_total.loc[(loc_501['start_datum']>='2019-10-19' )& (loc_501['start_datum']<='2019-10-27'),'vacation']=1
X_total.loc[loc_501['start_datum']>='2019-12-21','vacation']=1

X_train_option2=X_total.loc[X_total.index[loc_501['start_datum']<'2019-01-01'],:]
X_test_option2=X_total.loc[X_total.index[loc_501['start_datum']>'2018-12-31'],:]


y_501_train=y_total.loc[y_total.index[loc_501['start_datum']<'2019-01-01'],:].to_numpy().reshape((X_train_option2.shape[0],))
y_501_test=y_total.loc[y_total.index[loc_501['start_datum']>'2018-12-31'],:].to_numpy().reshape((X_test_option2.shape[0],))

# For first optimization, purely on sintime costime age and flow
X_train_501=pd.DataFrame({'sin_time':X_train_option2['sin_time'],'cos_time':X_train_option2['cos_time'],'mon':X_train_option2['mon'], 'tue':X_train_option2['tue'], 'wed':X_train_option2['wed'], 'thu':X_train_option2['thu'], 'fri':X_train_option2['fri'], 'sat':X_train_option2['sat'], 'sun':X_train_option2['sun'],'season1':X_train_option2['season'],'season2':X_train_option2['season2'],'feestdag':X_train_option2['feestdag'],'vacation':X_train_option2['vacation'],'Temperature':X_train_option2['Temperature'],'Rel_humidity':X_train_option2['Rel_humidity'],'Radiation':X_train_option2['Radiation']})
X_test_501=pd.DataFrame({'sin_time':X_test_option2['sin_time'],'cos_time':X_test_option2['cos_time'],'mon':X_test_option2['mon'], 'tue':X_test_option2['tue'], 'wed':X_test_option2['wed'], 'thu':X_test_option2['thu'], 'fri':X_test_option2['fri'], 'sat':X_test_option2['sat'], 'sun':X_test_option2['sun'],'season1':X_test_option2['season'],'season2':X_test_option2['season2'],'feestdag':X_test_option2['feestdag'],'vacation':X_test_option2['vacation'],'Temperature':X_test_option2['Temperature'],'Rel_humidity':X_test_option2['Rel_humidity'],'Radiation':X_test_option2['Radiation']})

#%% shuffel everything so better with validation 
np.random.seed(42)
shuffled_indices=np.random.permutation(len(X_train_501))
X_train_total_501_shuffled= X_train_501.loc[shuffled_indices,:].reset_index(drop=True)        

y_501_train_df=pd.DataFrame({'waarnemingen_intensiteit':y_501_train})
y_501_train_total_shuffled=y_501_train_df.loc[shuffled_indices,:].reset_index(drop=True) 

#%% train en validation set 
X_train_501_shuffled=X_train_total_501_shuffled[0:13612]
X_val_501_shuffled=X_train_total_501_shuffled[13612:]

y_501_train_shuffled=y_501_train_total_shuffled[0:13612]
y_501_val_shuffled=y_501_train_total_shuffled[13612:]

# %% Option 2 zorg dat alles geshuffled is
X_total=pd.concat([X_train_501,X_test_501])
y_train_df=pd.DataFrame({'true':y_501_train})
y_test_df=pd.DataFrame({'true':y_501_test})
y_total=pd.concat([y_train_df,y_test_df]).reset_index(drop=True)

np.random.seed(42)
shuffled_indices=np.random.permutation(len(X_total))

X_total_shuffled= X_total.loc[shuffled_indices,:].reset_index(drop=True)   
y_total_shuffled= y_total.loc[shuffled_indices,:].reset_index(drop=True)   

X_train_501_shuffled=X_total_shuffled.loc[0:13184,:]
X_val_501_shuffled=X_total_shuffled.loc[13184:16480,:]
X_test_501=X_total_shuffled.loc[16480:,:]

y_501_train_shuffled=y_total_shuffled.loc[0:13184,:]
y_501_val_shuffled=y_total_shuffled.loc[13184:16480,:]
y_501_test=y_total_shuffled.loc[16480:,:]



#%% BAyesian hyperparameter optimization
# 1. Make hyperparameter space

space={
    'n_estimators':hp.choice('n_estimators',[200,300,400,500,600]),
    'max_depth': hp.choice("max_depth",[ 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 20]),
    'min_samples_leaf':hp.choice('min_samples_leaf',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 20,21,22,23,24,25,26,27,28,29, 30])}
    

# 2. Build model
def build_rf_opt(params):
    randomforest_model=RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'])
    # randomforest_model=RandomForestRegressor(n_estimators=300, max_depth=params['max_depth'],min_samples_leaf=params['min_samples_leaf'])

    randomforest_model.fit(X_train_501_shuffled,y_501_train_shuffled)
    
    # predict on validation set
    y_val_between=randomforest_model.predict(X_val_501_shuffled)    
    y_val_between.shape=(y_val_between.shape[0],1)
    
    # calculate rmse on validation set
    forest_mse_val=mean_squared_error(y_val_between,y_501_val_shuffled)
    forest_rmse_val=np.sqrt(forest_mse_val)    
        
    print('Best validation loss:', forest_rmse_val)

    return {'loss': forest_rmse_val,   
            'status': STATUS_OK,
            'model': randomforest_model, 
            'params': params}

# 3. Pass the model to the optimization

trials =Trials() #saves everything
best=fmin(fn=build_rf_opt,
          space=space,
          algo=tpe.suggest,
          max_evals=100, # maybe 50 or so, just small to try now
          trials=trials
          )


# 4. evaluate
best_model = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['model']
best_params = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['params']
worst_model = trials.results[np.argmax([r['loss'] for r in 
    trials.results])]['model']
worst_params = trials.results[np.argmax([r['loss'] for r in 
    trials.results])]['params']

# %%final model

n_est=400#best_params['n_estimators'] 
max_d=10#best_params['max_depth']
min_l=8#best_params['min_samples_leaf']+1

# first define model
forest_reg=RandomForestRegressor(n_estimators=n_est, max_depth=max_d,min_samples_leaf=min_l)
forest_reg.fit(X_train_501_shuffled,y_501_train_shuffled)

# RESULTS
# predict on training set
forest_predict_train=forest_reg.predict(X_train_501_shuffled)
forest_predict_train.shape=(forest_predict_train.shape[0],1)

# predict on validation set
forest_predict_val=forest_reg.predict(X_val_501_shuffled)
forest_predict_val.shape=(forest_predict_val.shape[0],1)

# predict on test set
forest_predict_test=forest_reg.predict(X_test_501)
forest_predict_test.shape=(forest_predict_test.shape[0],1)

# change back 
forest_predict_train_back=(forest_predict_train*y_501_std+y_501_mean)[:,0]
forest_predict_val_back=(forest_predict_val*y_501_std+y_501_mean)[:,0]
forest_predict_test_back=(forest_predict_test*y_501_std+y_501_mean)[:,0]


y_501_train_back=(y_501_train_shuffled*y_501_std+y_501_mean)[:]
y_501_val_back=(y_501_val_shuffled*y_501_std+y_501_mean)[:]
y_501_test_back=(y_501_test*y_501_std+y_501_mean)[:]

# calculate performance
forest_mse_train=mean_squared_error(y_501_train_back,forest_predict_train_back)
forest_rmse_train=np.sqrt(forest_mse_train)
forest_mae_train=mean_absolute_error(y_501_train_back,forest_predict_train_back)

# on validation set
forest_mse_val=mean_squared_error(y_501_val_back,forest_predict_val_back)
forest_rmse_val=np.sqrt(forest_mse_val)
forest_mae_val=mean_absolute_error(y_501_val_back,forest_predict_val_back)

# evaluate on test set
forest_mse_test=mean_squared_error(y_501_test_back,forest_predict_test_back)
forest_rmse_test=np.sqrt(forest_mse_test)
forest_mae_test=mean_absolute_error(y_501_test_back,forest_predict_test_back)

print('forest rmse train set:',forest_rmse_train)
print('forest rmse test set',forest_rmse_test)
print('forest rmse val set',forest_rmse_val)

print('forest mae train set:',forest_mae_train)
print('forest mae val set:',forest_mae_val)
print('forest mae test set',forest_mae_test)

print('delta',(forest_rmse_val-forest_rmse_train)/forest_rmse_train*100)


#%% Feature importance
# je kan de importance van elke feature plotten.
for name,score in zip(X_train_501_shuffled.columns,forest_reg.feature_importances_):
    print(name,score)

#%% Make plots feature importance

names=['Time$_{sin}$','Time$_{cos}$','Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun','Season$_{sin}$','Season$_{cos}$','Free ','Vac','Temp','Hum','Rad']
names.reverse()
values=forest_reg.feature_importances_
color_list=[siemens_groen,siemens_groen_light3,siemens_groen_light1,siemens_groen_light3,siemens_groen_light3,siemens_groen_light3,siemens_groen_light1,siemens_groen_light1,siemens_groen_light3,siemens_groen_light3,siemens_groen_light3,siemens_groen_light3,siemens_groen_light3,siemens_groen_light3,siemens_groen,siemens_groen]

plt.figure()
plt.barh(names,values[::-1],color=color_list) # values[::-1] reverses it so start with sin time on top
plt.title('Relative feature importance of the random forest for location 501')
plt.xlabel('Relative feature importance')
plt.xlim((0,0.7))

#%% Extract single tree
# Extract single tree
for i in range(30):
    estimator = forest_reg.estimators_[i+200]
    
    text_rep=tree.export_text(estimator)
    
    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(estimator,max_depth=4, 
                       feature_names=X_501_train.columns,  
                       filled=True)


#%% Investigate single tree decision tree

max_d=10#best_params['max_depth']
min_l=20#best_params['min_samples_leaf']+1

tree_reg=DecisionTreeRegressor(max_depth=max_d,min_samples_leaf=min_l)
tree_reg.fit(X_train_501_shuffled,y_501_train_shuffled)
        
#predict on training set
tree_predict_train=tree_reg.predict(X_train_501_shuffled)
tree_predict_train.shape=(tree_predict_train.shape[0],1)

#predict on validation set
tree_predict_val=tree_reg.predict(X_val_501_shuffled)
tree_predict_val.shape=(tree_predict_val.shape[0],1)

# transform back
tree_predict_train_back=(tree_predict_train*y_501_std+y_501_mean)[:,0]
tree_predict_val_back=(tree_predict_val*y_501_std+y_501_mean)[:,0]

# true values back
y_501_train_back=(y_501_train_shuffled*y_501_std+y_501_mean)[:]
y_501_val_back=(y_501_val_shuffled*y_501_std+y_501_mean)[:]

# Evaluate
tree_mse_train=mean_squared_error(y_501_train_back,tree_predict_train_back)
tree_rmse_train=np.sqrt(tree_mse_train)
tree_mse_val=mean_squared_error(y_501_val_back,tree_predict_val_back)
tree_rmse_val=np.sqrt(tree_mse_val)

print('tree_rmse_train',tree_rmse_train) 
print('tree_rmse_val',tree_rmse_val) 

# print tree    

text_rep=tree.export_text(tree_reg)
print(text_rep)
fig = plt.figure(figsize=(25,20))
# plot tree
_ = tree.plot_tree(tree_reg, max_depth=2, feature_names=X_train_501_shuffled.columns,filled=True)
plt.title('First splits of the decision tree for location 531',fontsize=30)


