# This file makes the MLP

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

from tensorflow import keras

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
from sklearn.model_selection import cross_val_score #for cross validation
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# %% Define colors
color_tu=(0, 166/255, 214/255)
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
original_index_501=pd.read_pickle("C:/UserData/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_501.pkl")

def import_data_location(location):
    loc=pd.read_pickle("C:/UserData/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_{}.pkl".format(location))
    X_train=pd.read_pickle("C:/UserData/Documents/Afstuderen/python/Final_Feature_Set/X_{}_train_final.pkl".format(location))
    X_test=pd.read_pickle("C:/UserData/Documents/Afstuderen/python/Final_Feature_Set/X_{}_test_final.pkl".format(location))
    y_train=pd.read_pickle("C:/UserData/Documents/Afstuderen/python/Final_Feature_Set/y_{}_train.pkl".format(location))
    y_test=pd.read_pickle("C:/UserData/Documents/Afstuderen/python/Final_Feature_Set/y_{}_test.pkl".format(location))
    #info about y to scale back output and train on standardized set
    y_train_stand_values=stats.zscore(y_train.values).reshape(y_train.shape[0],)
    y_train_norm=(y_train-y_train.min())/(y_train.max()-y_train.min()) #min max normalization
    y_train_stand=pd.DataFrame({'waarnemingen_intensiteit':y_train_stand_values})
    y_train_mean=np.mean(y_train.values)
    y_train_std=np.std(y_train.values)
    y_test_stand=(y_test-y_train_mean)/y_train_std

    return loc,X_train,X_test,y_train,y_test,y_train_stand,y_test_stand,y_train_norm,y_train_mean,y_train_std

#Set up of the final feature set
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


# 2. split the set again in test and train
X_total=pd.concat([X_501_train_time,X_501_test_time])
X_total['season2']=season_nodig
y_total=pd.concat([y_501_train_stand,y_501_test_stand])

# Step 3 add school holiday as a feature
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

# %%Shuffle data and divide
np.random.seed(42)
shuffled_indices=np.random.permutation(len(X_train_501))
X_train_501_shuffled_total= X_train_501.loc[shuffled_indices,:].reset_index(drop=True)        
y_501_train_df=pd.DataFrame({'waarnemingen_intensiteit':y_501_train})
y_501_train_shuffled_total=y_501_train_df.loc[shuffled_indices,:].reset_index(drop=True) 

#train en validation set 
X_train_501_shuffled=X_train_501_shuffled_total[0:13612]
X_val_501_shuffled=X_train_501_shuffled_total[13612:]

y_501_train_shuffled=y_501_train_shuffled_total[0:13612]
y_501_val_shuffled=y_501_train_shuffled_total[13612:]


#%% Bayesian hyperparameter optimization

# 1. Make hyperparameter space

space={
    'n_hidden':hp.choice('n_hidden', [1,2,3,4,5]),
    'n_neurons': hp.choice('n_neurons', [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, 20,21,22,23,24,25,26,27,28,29, 30,31,32,33,34,35,36,37,38,39, 40, 41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]),
    'learning_rate': hp.choice('learning_rate', [0.00001,0.0001,0.001]) ,
    'batch_size':hp.choice('batch_size',[16,32,64])
    }

# 2. Build model
def build_mlp_opt(params):

    input_=keras.layers.Input(shape=X_train_501_shuffled_total.shape[1:])
    
    if params['n_hidden'] ==1:
        hidden1=keras.layers.Dense(params['n_neurons'],activation='relu')(input_)
        concat=keras.layers.Concatenate()([input_,hidden1])
        output=keras.layers.Dense(1)(concat)
        
    if params['n_hidden']  == 2:
        hidden1=keras.layers.Dense(params['n_neurons'],activation='relu')(input_)
        hidden2=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden1)
        concat=keras.layers.Concatenate()([input_,hidden2])
        output=keras.layers.Dense(1)(concat)        
        
    if params['n_hidden']  == 3:
        hidden1=keras.layers.Dense(params['n_neurons'],activation='relu')(input_)
        hidden2=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden1)
        hidden3=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden2)
        concat=keras.layers.Concatenate()([input_,hidden3])
        output=keras.layers.Dense(1)(concat)        
        
    if params['n_hidden']  == 4:
        hidden1=keras.layers.Dense(params['n_neurons'],activation='relu')(input_)
        hidden2=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden1)
        hidden3=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden2)
        hidden4=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden3)
        concat=keras.layers.Concatenate()([input_,hidden4])
        output=keras.layers.Dense(1)(concat)        
    
    if params['n_hidden']  == 5:
        hidden1=keras.layers.Dense(params['n_neurons'],activation='relu')(input_)
        hidden2=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden1)
        hidden3=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden2)
        hidden4=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden3)
        hidden5=keras.layers.Dense(params['n_neurons'],activation='relu')(hidden4)
        concat=keras.layers.Concatenate()([input_,hidden5])
        output=keras.layers.Dense(1)(concat)               
 
    
    model_mlp=keras.Model(inputs=[input_],outputs=[output])
    Optimizer=keras.optimizers.Adam(lr=params['learning_rate'])
    model_mlp.compile(loss='mse',optimizer=Optimizer)
    
    es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)    
    
    result=model_mlp.fit(X_train_501_shuffled_total,y_501_train_shuffled_total,epochs=60,validation_split=0.2,batch_size=params['batch_size'],callbacks=[es])

    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    
    return {'loss': validation_loss,   
            'status': STATUS_OK, 
            'model': model_mlp, 
            'params': params}


# 3. Pass the model to the optimization

trials =Trials() #saves everything
best=fmin(fn=build_mlp_opt,
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

#%% Make final nn

b_size=16
lrate=0.0001
n_neurons=55

input_=keras.layers.Input(shape=X_train_501_shuffled.shape[1:])
hidden1=keras.layers.Dense(n_neurons,activation='relu')(input_)
hidden2=keras.layers.Dense(n_neurons,activation='relu')(hidden1)
#hidden3=keras.layers.Dense(n_neurons,activation='relu')(hidden2) #If desired to have more layers
#hidden4=keras.layers.Dense(n_neurons,activation='relu')(hidden3)
#hidden5=keras.layers.Dense(n_neurons,activation='relu')(hidden4)

concat=keras.layers.Concatenate()([input_,hidden2])
output=keras.layers.Dense(1)(concat)        

model_mlp=keras.Model(inputs=[input_],outputs=[output])
Optimizer=keras.optimizers.Adam(lr=lrate)
model_mlp.compile(loss='mse',optimizer=Optimizer)
        
result=model_mlp.fit(X_train_501_shuffled,y_501_train_shuffled,epochs=60,validation_data=(X_val_501_shuffled,y_501_val_shuffled),batch_size=b_size)#,callbacks=[es])
   

# plot learning curve 
plt.figure()
plt.plot(result.history['loss'],label='training loss',color=siemens_groen)
plt.plot(result.history['val_loss'],label='validation loss',color=o1)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MSE [$veh^2/h^2$]')
plt.title("Learning curves ")
plt.ylim((0,0.8))

# %%RESULTS

# predict on training set
mlp_predict_train=model_mlp.predict(X_train_501_shuffled)
mlp_predict_train.shape=(mlp_predict_train.shape[0],1)

# predict on validation set
mlp_predict_val=model_mlp.predict(X_val_501_shuffled)
mlp_predict_val.shape=(mlp_predict_val.shape[0],1)

# predict on test set
mlp_predict_test=model_mlp.predict(X_test_501)
mlp_predict_test.shape=(mlp_predict_test.shape[0],1)

# change back 
mlp_predict_train_back=(mlp_predict_train*y_501_std+y_501_mean)[:,0]
mlp_predict_val_back=(mlp_predict_val*y_501_std+y_501_mean)[:,0]
mlp_predict_test_back=(mlp_predict_test*y_501_std+y_501_mean)[:,0]

y_501_train_back=(y_501_train_shuffled*y_501_std+y_501_mean)[:]
y_501_val_back=(y_501_val_shuffled*y_501_std+y_501_mean)[:]
y_501_test_back=(y_501_test*y_501_std+y_501_mean)[:]

# calculate performance
mlp_mse_train=mean_squared_error(y_501_train_back,mlp_predict_train_back)
mlp_rmse_train=np.sqrt(mlp_mse_train)
mlp_mae_train=mean_absolute_error(y_501_train_back,mlp_predict_train_back)

# evaluate on test set
mlp_mse_val=mean_squared_error(y_501_val_back,mlp_predict_val_back)
mlp_rmse_val=np.sqrt(mlp_mse_val)
mlp_mae_val=mean_absolute_error(y_501_val_back,mlp_predict_val_back)

# evaluate on test set
mlp_mse_test=mean_squared_error(y_501_test_back,mlp_predict_test_back)
mlp_rmse_test=np.sqrt(mlp_mse_test)
mlp_mae_test=mean_absolute_error(y_501_test_back,mlp_predict_test_back)

print('forest rmse train set:',mlp_rmse_train)
print('forest rmse test set',mlp_rmse_test)
print('forest rmse val set:',mlp_rmse_val)
print(10*'*')
print('forest mae train set:',mlp_mae_train)
print('forest mae val set',mlp_mae_val)
print('forest mae test set',mlp_mae_test)
print(10*'*')
print('delta',(mlp_rmse_val-mlp_rmse_train)/mlp_rmse_train*100)


# %% 2. OPTIMIZE LR
learning_rate=10**(-3)
mse_train_lr=[]
b_size=16
n_neurons=40 #63

for i in range(300):
   input_=keras.layers.Input(shape=X_train_501_shuffled.shape[1:])
   hidden1=keras.layers.Dense(n_neurons,activation='relu')(input_)
   hidden2=keras.layers.Dense(n_neurons,activation='relu')(hidden1)
   concat=keras.layers.Concatenate()([input_,hidden2])
   output=keras.layers.Dense(1)(concat)  
    
   model_mlp=keras.Model(inputs=[input_],outputs=[output])
   Optimizer=keras.optimizers.Adam(lr=learning_rate)
   model_mlp.compile(loss='mse',optimizer=Optimizer)
    
   result=model_mlp.fit(X_train_501_shuffled,y_501_train_shuffled,epochs=60,validation_split=0.2,batch_size=b_size)#,callbacks=[es])
     
   mse_test=model_mlp.evaluate(X_train_501_shuffled,y_501_train_shuffled)

   learning_rate=learning_rate*np.exp(np.log(10**6)/500)
   mse_train_lr=np.append(mse_train_lr,mse_test)

mse_test_lr_dropped=mse_train_lr[~np.isnan(mse_train_lr)]

