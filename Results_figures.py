# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:16:09 2022

@author: z0049unj
"""

# USed to make all figures

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import calplot
from matplotlib import font_manager as fm
import seaborn as sns

# colors
siemens_groen=(0/255,153/255,153/255)
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
siemens_groen_light1=(92/255,176/255,175/255)
siemens_groen_light2=(142/255,198/255,197/255)
siemens_groen_light3=(188/255,221/255,220/255)
siemens_blauw=(1/255,8/255,46/255)
siemens_blauw_groen=(0/255,90/255,120/255)
wit=(1,1,1)

#%% Import data
# Results and comparison figures

# obtain all predictions made diferent models  
# Final prediction transformer location 501
trans_y_pred_val_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\dropout02\y_pred_val_back_d02.csv',index_col=[0]).to_numpy()
trans_y_true_val_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\dropout02\y_true_val_back_d02.csv',index_col=[0]).to_numpy()
trans_y_pred_train_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\dropout02\y_pred_train_back_d02.csv',index_col=[0]).to_numpy()
trans_y_true_train_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\dropout02\y_true_train_back_d02.csv',index_col=[0]).to_numpy()
trans_y_pred_test_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\dropout02\y_pred_test_back_d02.csv',index_col=[0]).to_numpy()
trans_y_true_test_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\dropout02\y_true_test_back_d02.csv',index_col=[0]).to_numpy()
        
# Final prediction transformer location 531
trans_y_pred_val_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\dropout01\y_pred_val_back_d01.csv',index_col=[0]).to_numpy()
trans_y_true_val_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\dropout01\y_true_val_back_d01.csv',index_col=[0]).to_numpy()
trans_y_pred_train_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\dropout01\y_pred_train_back_d01.csv',index_col=[0]).to_numpy()
trans_y_true_train_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\dropout01\y_true_train_back_d01.csv',index_col=[0]).to_numpy()
trans_y_pred_test_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\dropout01\y_pred_test_back_d01.csv',index_col=[0]).to_numpy()
trans_y_true_test_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\dropout01\y_true_test_back_d01.csv',index_col=[0]).to_numpy()
           
# Simplified final prediction transformer location 501
trans_y_pred_val_501_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\simplified\y_pred_val_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_val_501_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\simplified\y_true_val_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_pred_train_501_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\simplified\y_pred_train_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_train_501_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\simplified\y_true_train_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_pred_test_501_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\simplified\y_pred_test_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_test_501_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\simplified\y_true_test_back_simplified.csv',index_col=[0]).to_numpy()
        
# Simplified final prediction transformer location 531
trans_y_pred_val_531_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\simplified\y_pred_val_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_val_531_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\simplified\y_true_val_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_pred_train_531_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\simplified\y_pred_train_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_train_531_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\simplified\y_true_train_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_pred_test_531_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\simplified\y_pred_test_back_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_test_531_simplified=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\simplified\y_true_test_back_simplified.csv',index_col=[0]).to_numpy()


 # 5th week as validation set final prediction transformer location 531
trans_y_pred_val_531_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\y_pred_val_back.csv',index_col=[0]).to_numpy()
trans_y_true_val_531_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\y_true_val_back.csv',index_col=[0]).to_numpy()
trans_y_pred_train_531_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\y_pred_train_back.csv',index_col=[0]).to_numpy()
trans_y_true_train_531_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\y_true_train_back.csv',index_col=[0]).to_numpy()
trans_y_pred_test_531_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\y_pred_test_back.csv',index_col=[0]).to_numpy()
trans_y_true_test_531_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\y_true_test_back.csv',index_col=[0]).to_numpy()
            
trans_y_pred_val_531_5th_sim=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\simplified\y_pred_val_back_5th_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_val_531_5th_sim=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\simplified\y_true_val_back_5th_simplified.csv',index_col=[0]).to_numpy()
trans_y_pred_train_531_5th_sim=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\simplified\y_pred_train_back_5th_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_train_531_5th_sim=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\simplified\y_true_train_back_5th_simplified.csv',index_col=[0]).to_numpy()
trans_y_pred_test_531_5th_sim=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\simplified\y_pred_test_back_5th_simplified.csv',index_col=[0]).to_numpy()
trans_y_true_test_531_5th_sim=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\5eweek_val_data\simplified\y_true_test_back_5th_simplified.csv',index_col=[0]).to_numpy()
                     
# Final prediction rf location 501
rf_y_pred_val_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\y_pred_val.csv',index_col=[0]).to_numpy()
rf_y_true_val_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\y_true_val.csv',index_col=[0]).to_numpy()
rf_y_pred_train_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\y_pred_train.csv',index_col=[0]).to_numpy()
rf_y_true_train_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\y_true_train.csv',index_col=[0]).to_numpy()
rf_y_pred_test_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\y_pred_test.csv',index_col=[0]).to_numpy()
rf_y_true_test_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\y_true_test.csv',index_col=[0]).to_numpy()
    
# Final prediction rf location 501
rf_y_pred_val_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\y_pred_val.csv',index_col=[0]).to_numpy()
rf_y_true_val_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\y_true_val.csv',index_col=[0]).to_numpy()
rf_y_pred_train_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\y_pred_train.csv',index_col=[0]).to_numpy()
rf_y_true_train_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\y_true_train.csv',index_col=[0]).to_numpy()
rf_y_pred_test_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\y_pred_test.csv',index_col=[0]).to_numpy()
rf_y_true_test_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\y_true_test.csv',index_col=[0]).to_numpy()
    
# Final prediction mlp location 501
mlp_y_pred_val_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\y_pred_val.csv',index_col=[0]).to_numpy()
mlp_y_true_val_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\y_true_val.csv',index_col=[0]).to_numpy()
mlp_y_pred_train_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\y_pred_train.csv',index_col=[0]).to_numpy()
mlp_y_true_train_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\y_true_train.csv',index_col=[0]).to_numpy()
mlp_y_pred_test_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\y_pred_test.csv',index_col=[0]).to_numpy()
mlp_y_true_test_501=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\y_true_test.csv',index_col=[0]).to_numpy()
    
# Final prediction mlp location 501
mlp_y_pred_val_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\y_pred_val.csv',index_col=[0]).to_numpy()
mlp_y_true_val_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\y_true_val.csv',index_col=[0]).to_numpy()
mlp_y_pred_train_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\y_pred_train.csv',index_col=[0]).to_numpy()
mlp_y_true_train_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\y_true_train.csv',index_col=[0]).to_numpy()
mlp_y_pred_test_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\y_pred_test.csv',index_col=[0]).to_numpy()
mlp_y_true_test_531=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\y_true_test.csv',index_col=[0]).to_numpy()
    
loc_501=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_501.pkl")
loc_531=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_531.pkl")

#%% load for extra figures, shuffled data and 5th week as validation
# 5th week as validation set final prediction transformer location 501
trans_y_pred_val_501_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\5eweek_val_data\y_pred_val_back_5th.csv',index_col=[0]).to_numpy()
trans_y_true_val_501_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\5eweek_val_data\y_true_val_back_5th.csv',index_col=[0]).to_numpy()
trans_y_pred_train_501_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\5eweek_val_data\y_pred_train_back_5th.csv',index_col=[0]).to_numpy()
trans_y_true_train_501_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\5eweek_val_data\y_true_train_back_5th.csv',index_col=[0]).to_numpy()
trans_y_pred_test_501_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\5eweek_val_data\y_pred_test_back_5th.csv',index_col=[0]).to_numpy()
trans_y_true_test_501_5th=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 501\5eweek_val_data\y_true_test_back_5th.csv',index_col=[0]).to_numpy()
        
# Final prediction rf location 501
rf_y_pred_val_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\shuffled_data\y_pred_val.csv',index_col=[0]).to_numpy()
rf_y_true_val_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\shuffled_data\y_true_val.csv',index_col=[0]).to_numpy()
rf_y_pred_train_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\shuffled_data\y_pred_train.csv',index_col=[0]).to_numpy()
rf_y_true_train_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\shuffled_data\y_true_train.csv',index_col=[0]).to_numpy()
rf_y_pred_test_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\shuffled_data\y_pred_test.csv',index_col=[0]).to_numpy()
rf_y_true_test_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\rf\shuffled_data\y_true_test.csv',index_col=[0]).to_numpy()
             
# Final prediction mlp location 501
mlp_y_pred_val_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\shuffled_data\y_pred_val.csv',index_col=[0]).to_numpy()
mlp_y_true_val_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\shuffled_data\y_true_val.csv',index_col=[0]).to_numpy()
mlp_y_pred_train_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\shuffled_data\y_pred_train.csv',index_col=[0]).to_numpy()
mlp_y_true_train_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\shuffled_data\y_true_train.csv',index_col=[0]).to_numpy()
mlp_y_pred_test_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\shuffled_data\y_pred_test.csv',index_col=[0]).to_numpy()
mlp_y_true_test_501_shuffle=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 501\mlp\shuffled_data\y_true_test.csv',index_col=[0]).to_numpy()
    

#%% Load for without march what is performance?
trans_y_pred_val_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\nomarch\y_pred_val_back.csv',index_col=[0]).to_numpy()
trans_y_true_val_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\nomarch\y_true_val_back.csv',index_col=[0]).to_numpy()
trans_y_pred_train_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\nomarch\y_pred_train_back.csv',index_col=[0]).to_numpy()
trans_y_true_train_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\nomarch\y_true_train_back.csv',index_col=[0]).to_numpy()
trans_y_pred_test_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\nomarch\y_pred_test_back.csv',index_col=[0]).to_numpy()
trans_y_true_test_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\TRANSFORMERS\Locatie 531\nomarch\y_true_test_back.csv',index_col=[0]).to_numpy()
        
# Final prediction rf location 501
rf_y_pred_val_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\nomarch\y_pred_val.csv',index_col=[0]).to_numpy()
rf_y_true_val_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\nomarch\y_true_val.csv',index_col=[0]).to_numpy()
rf_y_pred_train_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\nomarch\y_pred_train.csv',index_col=[0]).to_numpy()
rf_y_true_train_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\nomarch\y_true_train.csv',index_col=[0]).to_numpy()
rf_y_pred_test_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\nomarch\y_pred_test.csv',index_col=[0]).to_numpy()
rf_y_true_test_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\rf\nomarch\y_true_test.csv',index_col=[0]).to_numpy()
             
# Final prediction mlp location 501
mlp_y_pred_val_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\nomarch\y_pred_val.csv',index_col=[0]).to_numpy()
mlp_y_true_val_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\nomarch\y_true_val.csv',index_col=[0]).to_numpy()
mlp_y_pred_train_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\nomarch\y_pred_train.csv',index_col=[0]).to_numpy()
mlp_y_true_train_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\nomarch\y_true_train.csv',index_col=[0]).to_numpy()
mlp_y_pred_test_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\nomarch\y_pred_test.csv',index_col=[0]).to_numpy()
mlp_y_true_test_531_nomarch=pd.read_csv(r'C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\prediction_baseline\Locatie 531\mlp\nomarch\y_true_test.csv',index_col=[0]).to_numpy()

#%% Make figure of rmse en mae train val test set of all 

#RMSE for rf
RMSE_501_train_rf=np.sqrt(mean_squared_error(rf_y_pred_train_501,rf_y_true_train_501))
RMSE_501_val_rf=np.sqrt(mean_squared_error(rf_y_pred_val_501,rf_y_true_val_501))
RMSE_501_test_rf=np.sqrt(mean_squared_error(rf_y_pred_test_501,rf_y_true_test_501))

RMSE_531_train_rf=np.sqrt(mean_squared_error(rf_y_pred_train_531,rf_y_true_train_531))
RMSE_531_val_rf=np.sqrt(mean_squared_error(rf_y_pred_val_531,rf_y_true_val_531))
RMSE_531_test_rf=np.sqrt(mean_squared_error(rf_y_pred_test_531,rf_y_true_test_531))

#MAE for rf
MAE_501_train_rf=mean_absolute_error(rf_y_pred_train_501,rf_y_true_train_501)
MAE_501_val_rf=mean_absolute_error(rf_y_pred_val_501,rf_y_true_val_501)
MAE_501_test_rf=mean_absolute_error(rf_y_pred_test_501,rf_y_true_test_501)

MAE_531_train_rf=mean_absolute_error(rf_y_pred_train_531,rf_y_true_train_531)
MAE_531_val_rf=mean_absolute_error(rf_y_pred_val_531,rf_y_true_val_531)
MAE_531_test_rf=mean_absolute_error(rf_y_pred_test_531,rf_y_true_test_531)

#RMSE for mlp
RMSE_501_train_mlp=np.sqrt(mean_squared_error(mlp_y_pred_train_501,mlp_y_true_train_501))
RMSE_501_val_mlp=np.sqrt(mean_squared_error(mlp_y_pred_val_501,mlp_y_true_val_501))
RMSE_501_test_mlp=np.sqrt(mean_squared_error(mlp_y_pred_test_501,mlp_y_true_test_501))

RMSE_531_train_mlp=np.sqrt(mean_squared_error(mlp_y_pred_train_531,mlp_y_true_train_531))
RMSE_531_val_mlp=np.sqrt(mean_squared_error(mlp_y_pred_val_531,mlp_y_true_val_531))
RMSE_531_test_mlp=np.sqrt(mean_squared_error(mlp_y_pred_test_531,mlp_y_true_test_531))

#MAE for mlp
MAE_501_train_mlp=mean_absolute_error(mlp_y_pred_train_501,mlp_y_true_train_501)
MAE_501_val_mlp=mean_absolute_error(mlp_y_pred_val_501,mlp_y_true_val_501)
MAE_501_test_mlp=mean_absolute_error(mlp_y_pred_test_501,mlp_y_true_test_501)

MAE_531_train_mlp=mean_absolute_error(mlp_y_pred_train_531,mlp_y_true_train_531)
MAE_531_val_mlp=mean_absolute_error(mlp_y_pred_val_531,mlp_y_true_val_531)
MAE_531_test_mlp=mean_absolute_error(mlp_y_pred_test_531,mlp_y_true_test_531)

# RMSE for transformer
RMSE_501_train_trans=np.sqrt(mean_squared_error(trans_y_pred_train_501,trans_y_true_train_501))
RMSE_501_val_trans=np.sqrt(mean_squared_error(trans_y_pred_val_501,trans_y_true_val_501))
RMSE_501_test_trans=np.sqrt(mean_squared_error(trans_y_pred_test_501,trans_y_true_test_501))

RMSE_501_train_trans_simplified=np.sqrt(mean_squared_error(trans_y_pred_train_501_simplified,trans_y_true_train_501_simplified))
RMSE_501_val_trans_simplified=np.sqrt(mean_squared_error(trans_y_pred_val_501_simplified,trans_y_true_val_501_simplified))
RMSE_501_test_trans_simplified=np.sqrt(mean_squared_error(trans_y_pred_test_501_simplified,trans_y_true_test_501_simplified))

RMSE_531_train_trans=np.sqrt(mean_squared_error(trans_y_pred_train_531,trans_y_true_train_531))
RMSE_531_val_trans=np.sqrt(mean_squared_error(trans_y_pred_val_531,trans_y_true_val_531))
RMSE_531_test_trans=np.sqrt(mean_squared_error(trans_y_pred_test_531,trans_y_true_test_531))

# MAE for transformer
MAE_501_train_trans=mean_absolute_error(trans_y_pred_train_501,trans_y_true_train_501)
MAE_501_val_trans=mean_absolute_error(trans_y_pred_val_501,trans_y_true_val_501)
MAE_501_test_trans=mean_absolute_error(trans_y_pred_test_501,trans_y_true_test_501)

MAE_531_train_trans=mean_absolute_error(trans_y_pred_train_531,trans_y_true_train_531)
MAE_531_val_trans=mean_absolute_error(trans_y_pred_val_531,trans_y_true_val_531)
MAE_531_test_trans=mean_absolute_error(trans_y_pred_test_531,trans_y_true_test_531)

#%%For extra figures 
# RMSE
RMSE_501_train_trans_5th=np.sqrt(mean_squared_error(trans_y_pred_train_501_5th,trans_y_true_train_501_5th))
RMSE_501_val_trans_5th=np.sqrt(mean_squared_error(trans_y_pred_val_501_5th,trans_y_true_val_501_5th))
RMSE_501_test_trans_5th=np.sqrt(mean_squared_error(trans_y_pred_test_501_5th,trans_y_true_test_501_5th))

RMSE_501_train_rf_shuffle=np.sqrt(mean_squared_error(rf_y_pred_train_501_shuffle,rf_y_true_train_501_shuffle))
RMSE_501_val_rf_shuffle=np.sqrt(mean_squared_error(rf_y_pred_val_501_shuffle,rf_y_true_val_501_shuffle))
RMSE_501_test_rf_shuffle=np.sqrt(mean_squared_error(rf_y_pred_test_501_shuffle,rf_y_true_test_501_shuffle))

RMSE_501_train_mlp_shuffle=np.sqrt(mean_squared_error(mlp_y_pred_train_501_shuffle,mlp_y_true_train_501_shuffle))
RMSE_501_val_mlp_shuffle=np.sqrt(mean_squared_error(mlp_y_pred_val_501_shuffle,mlp_y_true_val_501_shuffle))
RMSE_501_test_mlp_shuffle=np.sqrt(mean_squared_error(mlp_y_pred_test_501_shuffle,mlp_y_true_test_501_shuffle))

# RMSE
RMSE_531_train_trans_nomarch=np.sqrt(mean_squared_error(trans_y_pred_train_531_nomarch,trans_y_true_train_531_nomarch))
RMSE_531_val_trans_nomarch=np.sqrt(mean_squared_error(trans_y_pred_val_531_nomarch,trans_y_true_val_531_nomarch))
RMSE_531_test_trans_nomarch=np.sqrt(mean_squared_error(trans_y_pred_test_531_nomarch,trans_y_true_test_531_nomarch))

RMSE_531_train_rf_nomarch=np.sqrt(mean_squared_error(rf_y_pred_train_531_nomarch,rf_y_true_train_531_nomarch))
RMSE_531_val_rf_nomarch=np.sqrt(mean_squared_error(rf_y_pred_val_531_nomarch,rf_y_true_val_531_nomarch))
RMSE_531_test_rf_nomarch=np.sqrt(mean_squared_error(rf_y_pred_test_531_nomarch,rf_y_true_test_531_nomarch))

RMSE_531_train_mlp_nomarch=np.sqrt(mean_squared_error(mlp_y_pred_train_531_nomarch,mlp_y_true_train_531_nomarch))
RMSE_531_val_mlp_nomarch=np.sqrt(mean_squared_error(mlp_y_pred_val_531_nomarch,mlp_y_true_val_531_nomarch))
RMSE_531_test_mlp_nomarch=np.sqrt(mean_squared_error(mlp_y_pred_test_531_nomarch,mlp_y_true_test_531_nomarch))


#%% Locatie 501

# make a bargraph of these 
barWidth=0.9
bar_train=[RMSE_501_train_rf,RMSE_501_train_mlp,RMSE_501_train_trans]
bar_val=[RMSE_501_val_rf,RMSE_501_val_mlp,RMSE_501_val_trans]
bar_test=[RMSE_501_test_rf,RMSE_501_test_mlp,RMSE_501_test_trans]
#MAE
bar_train_mae=[MAE_501_train_rf,MAE_501_train_mlp,MAE_501_train_trans]
bar_val_mae=[MAE_501_val_rf,MAE_501_val_mlp,MAE_501_val_trans]
bar_test_mae=[MAE_501_test_rf,MAE_501_test_mlp,MAE_501_test_trans]


#x positions of the bars
r1 = [1,5,9]
r2 = [2,6,10]
r3 = [3,7,11]

plt.figure()
plt.bar(r1, bar_train, width = barWidth, color = siemens_blauw_groen, label='Train')
plt.bar(r2, bar_val, width = barWidth, color = siemens_groen, label='Validation')
plt.bar(r3, bar_test, width = barWidth, color = siemens_groen_light2, label='Test')
# Note: the barplot could be created easily. See the barplot section for other examples.
plt.ylabel('RMSE [veh/h]')
plt.xticks([2,6,10],['Random forest','Multilayer perceptron','Transformer'])
# Create legend
plt.title('RMSE for location 501')
plt.legend()
plt.ylim((0,175))

plt.figure()
plt.bar(r1, bar_train_mae, width = barWidth, color = o1, label='Train')
plt.bar(r2, bar_val_mae, width = barWidth, color = o4, label='Validation')
plt.bar(r3, bar_test_mae, width = barWidth, color = o7, label='Test')
# Note: the barplot could be created easily. See the barplot section for other examples.
plt.ylabel('MAE [veh/h]')
plt.xticks([2,6,10],['Random forest','Multilayer perceptron','Transformer'])
# Create legend
plt.title('MAE location 501')
plt.legend()
plt.ylim((0,175))

# Locatie 531
bar_train_531=[RMSE_531_train_rf,RMSE_531_train_mlp,RMSE_531_train_trans]
bar_val_531=[RMSE_531_val_rf,RMSE_531_val_mlp,RMSE_531_val_trans]
bar_test_531=[RMSE_531_test_rf,RMSE_531_test_mlp,RMSE_531_test_trans]
#MAE
bar_train_mae_531=[MAE_531_train_rf,MAE_531_train_mlp,MAE_531_train_trans]
bar_val_mae_531=[MAE_531_val_rf,MAE_531_val_mlp,MAE_531_val_trans]
bar_test_mae_531=[MAE_531_test_rf,MAE_531_test_mlp,MAE_531_test_trans]
              
plt.figure()       
plt.bar(r1, bar_train_531, width = barWidth, color = siemens_blauw_groen, label='Train')
plt.bar(r2, bar_val_531, width = barWidth, color = siemens_groen, label='Validation')
plt.bar(r3, bar_test_531, width = barWidth, color = siemens_groen_light2, label='Test')
# Note: the barplot could be created easily. See the barplot section for other examples.

plt.ylabel('RMSE [veh/h]')
plt.xticks([2,6,10],['Random forest','Multilayer perceptron','Transformer'])
# Create legend
plt.title('RMSE for location 531')
plt.legend()
plt.ylim((0,175))


plt.figure()
plt.bar(r1, bar_train_mae_531, width = barWidth, color = o1, label='Train')
plt.bar(r2, bar_val_mae_531, width = barWidth, color = o4, label='Validation')
plt.bar(r3, bar_test_mae_531, width = barWidth, color = o7, label='Test')
# Note: the barplot could be created easily. See the barplot section for other examples.
plt.ylabel('MAE [veh/h]')
plt.xticks([2,6,10],['Random forest','Multilayer perceptron','Transformer'])
# Create legend
plt.title('MAE for location 531')
plt.legend()
plt.ylim((0,175))

#%% Locatie 501 WITH EXTRA TRANSFORMER  bargraph

# make a bargraph of these 
barWidth=0.9

bar_train=[RMSE_501_train_rf,RMSE_501_train_rf_shuffle,RMSE_501_train_mlp,RMSE_501_train_mlp_shuffle,RMSE_501_train_trans,RMSE_501_train_trans_5th]
bar_val=[RMSE_501_val_rf,RMSE_501_val_rf_shuffle,RMSE_501_val_mlp,RMSE_501_val_mlp_shuffle,RMSE_501_val_trans,RMSE_501_val_trans_5th]
bar_test=[RMSE_501_test_rf,RMSE_501_test_rf_shuffle,RMSE_501_test_mlp,RMSE_501_test_mlp_shuffle,RMSE_501_test_trans,RMSE_501_test_trans_5th]


#x positions of the bars
r1 = [1,5,9,13,17,21]
r2 = [2,6,10,14,18,22]
r3 = [3,7,11,15,19,23]

plt.figure()
plt.bar(r1, bar_train, width = barWidth, color = siemens_blauw_groen, label='Train')
plt.bar(r2, bar_val, width = barWidth, color = siemens_groen, label='Validation')
plt.bar(r3, bar_test, width = barWidth, color = siemens_groen_light2, label='Test')


# Note: the barplot could be created easily. See the barplot section for other examples.
plt.ylabel('RMSE [veh/h]')
plt.xticks([2,6,10,14,18,22],['Random forest','Random forest 2','MLP','MLP 2','Transformer','Transformer 2'],rotation=20)
# Create legend
plt.title('RMSE for location 501')
plt.legend(loc=2)
plt.ylim((0,200))


#%% RMSE test set baseline models 
predictions=np.linspace(0,23,24).reshape((1,24))
RMSE_hour_rf_501=np.zeros((1,24))
RMSE_hour_rf_531=np.zeros((1,24))
RMSE_hour_mlp_501=np.zeros((1,24))
RMSE_hour_mlp_531=np.zeros((1,24))

MAE_hour_rf_501=np.zeros((1,24))
MAE_hour_rf_531=np.zeros((1,24))
MAE_hour_mlp_501=np.zeros((1,24))
MAE_hour_mlp_531=np.zeros((1,24))



for h in range(24): # for each start time
    y_pred_hour_rf_501=rf_y_pred_test_501[h::24,:]
    y_pred_hour_rf_531=rf_y_pred_test_531[h::24,:]
    y_pred_hour_mlp_501=mlp_y_pred_test_501[h::24,:]
    y_pred_hour_mlp_531=mlp_y_pred_test_531[h::24,:]
    
    y_true_hour_rf_501=rf_y_true_test_501[h::24,:]
    y_true_hour_rf_531=rf_y_true_test_531[h::24,:]
    y_true_hour_mlp_501=mlp_y_true_test_501[h::24,:]
    y_true_hour_mlp_531=mlp_y_true_test_531[h::24,:]    
    
    
    RMSE_rf_501=np.sqrt(mean_squared_error(y_pred_hour_rf_501,y_true_hour_rf_501))
    MAE_rf_501=mean_absolute_error(y_pred_hour_rf_501,y_true_hour_rf_501)

    RMSE_rf_531=np.sqrt(mean_squared_error(y_pred_hour_rf_531,y_true_hour_rf_531))
    MAE_rf_531=mean_absolute_error(y_pred_hour_rf_531,y_true_hour_rf_531)
    
    RMSE_mlp_501=np.sqrt(mean_squared_error(y_pred_hour_mlp_501,y_true_hour_mlp_501))
    MAE_mlp_501=mean_absolute_error(y_pred_hour_mlp_501,y_true_hour_mlp_501)

    RMSE_mlp_531=np.sqrt(mean_squared_error(y_pred_hour_mlp_531,y_true_hour_mlp_531))
    MAE_mlp_531=mean_absolute_error(y_pred_hour_mlp_531,y_true_hour_mlp_531)
    
    
    RMSE_hour_rf_501[0,h]=RMSE_rf_501
    MAE_hour_rf_501[0,h]=MAE_rf_501
    
    RMSE_hour_rf_531[0,h]=RMSE_rf_531
    MAE_hour_rf_531[0,h]=MAE_rf_531
    
    RMSE_hour_mlp_501[0,h]=RMSE_mlp_501
    MAE_hour_mlp_501[0,h]=MAE_mlp_501
    
    RMSE_hour_mlp_531[0,h]=RMSE_mlp_531
    MAE_hour_mlp_531[0,h]=MAE_mlp_531
    
    
cmap_2_colors=LinearSegmentedColormap.from_list("", [wit,o1])
y = np.arange(1)
RMSE_baseline_501=np.concatenate((RMSE_hour_rf_501,RMSE_hour_mlp_501))
my_xticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']


fig, ax = plt.subplots()
im=ax.imshow(RMSE_baseline_501,cmap=cmap_2_colors,vmin=0,vmax=250,aspect='equal')
cbar_ax = fig.add_axes([0.93, 0.26, 0.02, 0.3]) # [left, bottom, width, height]
fig.colorbar(im,cax=cbar_ax)

y_ticks=('Random forest','Multilayer perceptron')
y_loc=[0,1]
ax.set_yticks(np.linspace(0,1,2))
ax.set_yticklabels(y_ticks)
ax.set_xticks(np.linspace(0,23,24))
ax.set_xticklabels(my_xticks,rotation=80)#,fontsize=8)
ax.set_xlabel('Time of the day [hh:m:ss]')

ax.title.set_text('Average RMSE for location 501')

RMSE_baseline_531=np.concatenate((RMSE_hour_rf_531,RMSE_hour_mlp_531))
my_xticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']


fig, ax = plt.subplots()
im=ax.imshow(RMSE_baseline_531,cmap=cmap_2_colors,vmin=0,vmax=250,aspect='equal')
cbar_ax = fig.add_axes([0.93, 0.26, 0.02, 0.3]) # [left, bottom, width, height]
fig.colorbar(im,cax=cbar_ax)

y_ticks=('Random forest','Multilayer perceptron')
y_loc=[0,1]
ax.set_yticks(np.linspace(0,1,2))
ax.set_yticklabels(y_ticks)
ax.set_xticks(np.linspace(0,23,24))
ax.set_xticklabels(my_xticks,rotation=80)#,fontsize=8)
ax.set_xlabel('Time of the day [hh:m:ss]')

ax.title.set_text('Average RMSE for location 501')

#%%RMSE transformer multiple horizons and start time
RMSE_array_hour_501=np.zeros((24,24))
RMSE_array_hour_531=np.zeros((24,24))
MAE_array_hour_501=np.zeros((24,24))
MAE_array_hour_531=np.zeros((24,24))

RMSE_array_hour_501_simplified=np.zeros((24,24))
RMSE_array_hour_531_simplified=np.zeros((24,24))
MAE_array_hour_501_simplified=np.zeros((24,24))
MAE_array_hour_531_simplified=np.zeros((24,24))

for h in range(24): # for each start time
    y_pred_hour_trans_501=trans_y_pred_test_501[:,h]
    y_true_hour_trans_501=trans_y_true_test_501[:,h]
    y_pred_hour_trans_531=trans_y_pred_test_531[:,h]
    y_true_hour_trans_531=trans_y_true_test_531[:,h]
    
    y_pred_hour_trans_501_simplified=trans_y_pred_test_501_simplified[:,h]
    y_true_hour_trans_501_simplified=trans_y_true_test_501_simplified[:,h]
    y_pred_hour_trans_531_simplified=trans_y_pred_test_531_simplified[:,h]
    y_true_hour_trans_531_simplified=trans_y_true_test_531_simplified[:,h]
        
    for i in range(24):
        y_pred_horizon_501=y_pred_hour_trans_501[i::24]
        y_pred_horizon_531=y_pred_hour_trans_531[i::24]

        y_true_horizon_501=y_true_hour_trans_501[i::24]
        y_true_horizon_531=y_true_hour_trans_531[i::24]
       
        y_pred_horizon_501_simplified=y_pred_hour_trans_501_simplified[i::24]
        y_pred_horizon_531_simplified=y_pred_hour_trans_531_simplified[i::24]

        y_true_horizon_501_simplified=y_true_hour_trans_501_simplified[i::24]
        y_true_horizon_531_simplified=y_true_hour_trans_531_simplified[i::24]  
       
        RMSE_501=np.sqrt(mean_squared_error(y_pred_horizon_501,y_true_horizon_501))
        MAE_501=mean_absolute_error(y_pred_horizon_501,y_true_horizon_501)
        RMSE_531=np.sqrt(mean_squared_error(y_pred_horizon_531,y_true_horizon_531))
        MAE_531=mean_absolute_error(y_pred_horizon_531,y_true_horizon_531)

        RMSE_501_simplified=np.sqrt(mean_squared_error(y_pred_horizon_501_simplified,y_true_horizon_501_simplified))
        MAE_501_simplified=mean_absolute_error(y_pred_horizon_501_simplified,y_true_horizon_501_simplified)
        RMSE_531_simplified=np.sqrt(mean_squared_error(y_pred_horizon_531_simplified,y_true_horizon_531_simplified))
        MAE_531_simplified=mean_absolute_error(y_pred_horizon_531_simplified,y_true_horizon_531_simplified)
        
        a=i+h
        if a>23:
            a_final=a-24
        else:
            a_final=a
        RMSE_array_hour_501[a_final,h]=RMSE_501
        RMSE_array_hour_531[a_final,h]=RMSE_531
        MAE_array_hour_501[a_final,h]=MAE_501
        MAE_array_hour_531[a_final,h]=MAE_531
        
        RMSE_array_hour_501_simplified[a_final,h]=RMSE_501_simplified
        RMSE_array_hour_531_simplified[a_final,h]=RMSE_531_simplified
        MAE_array_hour_501_simplified[a_final,h]=MAE_501_simplified
        MAE_array_hour_531_simplified[a_final,h]=MAE_531_simplified
        
cmap_2_colors_rmse=LinearSegmentedColormap.from_list("", [wit,o1])
cmap_2_colors_mae=LinearSegmentedColormap.from_list("", [wit,o1])

#LOCATION 501 RMSE
fig,ax=plt.subplots()     
heatmap=ax.imshow(RMSE_array_hour_501,cmap=cmap_2_colors_rmse,vmin=0,vmax=350)
ax.title.set_text('RMSE location 501')

#colorbar
cbar=plt.colorbar(heatmap)
cbar.ax.set_ylabel('RMSE [veh/h]', rotation=(360-90))
cbar.ax.tick_params(labelsize=9)

# y labels
my_yticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']
ax.set_yticks(np.linspace(0,23,24))
ax.set_yticklabels(my_yticks,rotation=0,fontsize=9)
ax.set_ylabel('Time of the day [hh:m:ss]')

# x labels
ax.set_xlabel('Prediction horizon [h]')
my_xticks=['1','3','5','7','9','11','13','15','17','19','21','23']
ax.set_xticks(np.array([0,2,4,6,8,10,12,14,16,18,20,22]))
ax.set_xticklabels(my_xticks,rotation=0,fontsize=9)
ax.set_xlabel('Prediction horizon [h]')

#LOCATION 531 RMSE
fig,ax=plt.subplots()     
heatmap=ax.imshow(RMSE_array_hour_531,cmap=cmap_2_colors_rmse,vmin=0,vmax=350)
ax.title.set_text('RMSE location 531')

#colorbar
cbar=plt.colorbar(heatmap)
cbar.ax.set_ylabel('RMSE [veh/h]', rotation=(360-90))
cbar.ax.tick_params(labelsize=9)

# y labels
ax.set_ylabel('Start time of the day [hh:mm:ss]')
my_yticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']
ax.set_yticks(np.linspace(0,23,24))
ax.set_yticklabels(my_yticks,rotation=0,fontsize=9)
ax.set_ylabel('Time of the day [hh:m:ss]')

# x labels
ax.set_xlabel('Prediction horizon [h]')
my_xticks=['1','3','5','7','9','11','13','15','17','19','21','23']
ax.set_xticks(np.array([0,2,4,6,8,10,12,14,16,18,20,22]))
ax.set_xticklabels(my_xticks,rotation=0,fontsize=9)
ax.set_xlabel('Prediction horizon [h]')

#%% Plot for each time the error at different prediction horizons
prediction_horizons=np.linspace(1,24,24)

for i in range(24):
    
    RMSE_array_rf_501= np.full((24,),RMSE_hour_rf_501[0,i])
    RMSE_array_mlp_501= np.full((24,),RMSE_hour_mlp_501[0,i])
    MAE_array_rf_501= np.full((24,),MAE_hour_rf_501[0,i])
    MAE_array_mlp_501= np.full((24,),MAE_hour_mlp_501[0,i])    
    
    fig,ax=plt.subplots(1,2)
    ax[0].plot(prediction_horizons,RMSE_array_hour_501[i,:],label='Transformer',color=siemens_groen)
    ax[0].plot(prediction_horizons,RMSE_array_rf_501,label='Random forest',color=siemens_groen,linestyle='--')
    ax[0].plot(prediction_horizons,RMSE_array_mlp_501,label='Multilayer perceptron',color=siemens_groen,linestyle=':')
    ax[0].set_title('RMSE')

    ax[1].plot(prediction_horizons,MAE_array_hour_501[i,:],label='Transformer',color=siemens_groen)
    ax[1].plot(prediction_horizons,MAE_array_rf_501,label='Random forest',color=siemens_groen,linestyle='--')
    ax[1].plot(prediction_horizons,MAE_array_mlp_501,label='Multilayer perceptron',color=siemens_groen,linestyle=':')
    ax[1].set_title('MAE')
      
    ax[0].set_ylim((0,180))
    ax[1].set_ylim((0,180))
    plt.xlim((0.9,24))
    fig.suptitle('Performance for location 501 at time {}'.format(i))
        
    
#%%locatie 531
for i in range(24):
    
    RMSE_array_rf_531= np.full((24,),RMSE_hour_rf_531[0,i])
    RMSE_array_mlp_531= np.full((24,),RMSE_hour_mlp_531[0,i])
    MAE_array_rf_531= np.full((24,),MAE_hour_rf_531[0,i])
    MAE_array_mlp_531= np.full((24,),MAE_hour_mlp_531[0,i])    
    
    fig,ax=plt.subplots(1,2)
    ax[0].plot(prediction_horizons,RMSE_array_hour_531[i,:],label='Transformer',color=siemens_groen)
    ax[0].plot(prediction_horizons,RMSE_array_rf_531,label='Random forest',color=siemens_groen,linestyle='--')
    ax[0].plot(prediction_horizons,RMSE_array_mlp_531,label='Multilayer perceptron',color=siemens_groen,linestyle=':')
    ax[0].set_title('RMSE')

    ax[1].plot(prediction_horizons,MAE_array_hour_531[i,:],label='Transformer',color=siemens_groen)
    ax[1].plot(prediction_horizons,MAE_array_rf_531,label='Random forest',color=siemens_groen,linestyle='--')
    ax[1].plot(prediction_horizons,MAE_array_mlp_531,label='Multilayer perceptron',color=siemens_groen,linestyle=':')
    ax[1].set_title('MAE')
      
    ax[0].set_ylim((0,450))
    ax[1].set_ylim((0,450))

    plt.xlim((0.9,24))
    fig.suptitle('Performance for location 531 at time {}'.format(i))
        

#%% Make median figure with deviations, or maybe better cluster figure
my_xticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']

#Location 501
data501_pivot=loc_501.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
data_501_median=data501_pivot.median()
data_501_5=data501_pivot.quantile(q=0.05)
data_501_25=data501_pivot.quantile(q=0.25)
data_501_75=data501_pivot.quantile(q=0.75)
data_501_95=data501_pivot.quantile(q=0.95)

horizon=np.linspace(0,23,24)
ax =plt.figure()
ax=sns.lineplot(x=horizon, y=data_501_median,color=siemens_blauw,label='median')
ax.fill_between(horizon, data_501_5, data_501_95,color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(horizon, data_501_25, data_501_75,color=siemens_groen,label='25%-75%')
plt.legend(loc=2)
plt.title('Behavior of the prediction models location 501')
ax.set_xticks(np.linspace(0,23,24))
ax.set_xticklabels(my_xticks,rotation=60,fontsize=9)
ax.set_xlabel('Time of the day [hh:mm:ss]')
ax.set_ylabel('Traffic flow [veh/h]')
ax.set_ylim((0,2000))
ax.set_xlim((0,23))

plt.plot((0.5,0.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((10.5,10.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((13.5,13.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((22.5,22.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
ax.fill_between([0,0.5], [2000,2000],color=siemens_groen_light1,alpha=0.2)
ax.fill_between([0.5,10.5], [2000,2000],color=o1,alpha=0.2)
ax.fill_between([10.5,13.5], [2000,2000],color=siemens_groen_light1,alpha=0.2)
ax.fill_between([13.5,22.5], [2000,2000],color=siemens_groen,alpha=0.4)
ax.fill_between([22.5,23], [2000,2000],color=siemens_groen_light1,alpha=0.2)

#%%Location 531
data531_pivot=loc_531.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
data_531_median=data531_pivot.median()
data_531_5=data531_pivot.quantile(q=0.05)
data_531_25=data531_pivot.quantile(q=0.25)
data_531_75=data531_pivot.quantile(q=0.75)
data_531_95=data531_pivot.quantile(q=0.95)

horizon=np.linspace(0,23,24)
ax =plt.figure()
ax=sns.lineplot(x=horizon, y=data_531_median,color=siemens_blauw,label='median')
ax.fill_between(horizon, data_531_5, data_531_95,color='None',label='5%-95%',edgecolor=siemens_blauw,linestyle=':')
ax.fill_between(horizon, data_531_25, data_531_75,color='None',label='25%-75%',edgecolor=siemens_blauw,linestyle='--')
plt.legend(loc=2)
plt.title('Behavior of the prediction models location 531')
ax.set_xticks(np.linspace(0,23,24))
ax.set_xticklabels(my_xticks,rotation=60,fontsize=9)
ax.set_xlabel('Time of the day [hh:mm:ss]')
ax.set_ylabel('Traffic flow [veh/h]')
ax.set_ylim((0,2000))
ax.set_xlim((0,23))
plt.plot((5.5,5.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((7.5,7.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((9.5,9.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((14.5,14.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((16.5,16.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((17.5,17.5),(0,2000), linestyle=':',color='k',linewidth=0.7)
plt.plot((18.5,18.5),(0,2000), linestyle=':',color='k',linewidth=0.7)

ax.fill_between([0,5.5], [2000,2000],color=o1,alpha=0.2)
ax.fill_between([5.5,7.5], [2000,2000],color=siemens_groen_light1,alpha=0.2)
ax.fill_between([7.5,9.5], [2000,2000],color=siemens_groen,alpha=0.4)
ax.fill_between([9.5,14.5], [2000,2000],color=o1,alpha=0.2)
ax.fill_between([14.5,16.5], [2000,2000],color=siemens_groen,alpha=0.4)
ax.fill_between([16.5,17.5], [2000,2000],color='r',alpha=0.2)
ax.fill_between([17.5,18.5], [2000,2000],color=siemens_groen_light1,alpha=0.2)
ax.fill_between([18.5,23], [2000,2000],color=o1,alpha=0.2)

#%% Make  figure with all three model performances in one figure
# LOCATION 501
cmap_2_colors_rmse=LinearSegmentedColormap.from_list("", [wit,o1])

fig,ax=plt.subplots(nrows=1,ncols=3, gridspec_kw={'width_ratios': [ 1,1,24]})
fig.suptitle(' Performance on location 501 ', fontsize=12)

im_rf=ax[0].imshow(np.transpose(RMSE_hour_rf_501),cmap=cmap_2_colors_rmse,vmin=0,vmax=400)
im_mlp=ax[1].imshow(np.transpose(RMSE_hour_mlp_501),cmap=cmap_2_colors_rmse,vmin=0,vmax=400)
im_trans=ax[2].imshow(RMSE_array_hour_501,cmap=cmap_2_colors_rmse,vmin=0,vmax=400)

# set y axis plot 1 the time
ax[0].set_ylabel('Time of the day [hh:mm:ss]')
my_yticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']
ax[0].set_yticks(np.linspace(0,23,24))
ax[0].set_yticklabels(my_yticks,rotation=0,fontsize=9)

# remove y labels of plot 2 en 3
ax[1].set_yticklabels([0],'')#,fontsize=8)
ax[2].set_yticklabels([0],'')#,fontsize=8)

# remove x labels of plot 1 en 2
my_xticks_rf=['']
ax[0].set_xticks(np.array([0]))
ax[0].set_xticklabels(my_xticks_rf)#,fontsize=8)

my_xticks_mlp=['']
ax[1].set_xticks(np.array([0]))
ax[1].set_xticklabels(my_xticks_mlp)#,fontsize=8)

# set x labels of transformer plot
my_xticks_trans=['1','3','5','7','9','11','13','15','17','19','21','23']
ax[2].set_xticks(np.array([0,2,4,6,8,10,12,14,16,18,20,22]))
ax[2].set_xticklabels(my_xticks_trans)#,fontsize=8)
ax[2].set_xlabel('Prediction horizon [h]')#,fontsize=8)

#Set titles subplots
ax[0].set_title('RF',fontsize=10)
ax[1].set_title('MLP',fontsize=10)
ax[2].set_title('Transformer',fontsize=10)

# plot colorbar
cbar=fig.colorbar(im_trans)
cbar.ax.set_ylabel('RMSE [veh/h]', rotation=(360-90),fontsize=9,labelpad=10)

plt.subplots_adjust(left=0.1, bottom=0.1,right=0.9, top=0.86, wspace=0.1, hspace=0.1)

# LOCATION 531
fig,ax=plt.subplots(nrows=1,ncols=3, gridspec_kw={'width_ratios': [ 1,1,24]})
fig.suptitle(' Performance on location 531 ', fontsize=12)

im_rf=ax[0].imshow(np.transpose(RMSE_hour_rf_531),cmap=cmap_2_colors_rmse,vmin=0,vmax=400)
im_mlp=ax[1].imshow(np.transpose(RMSE_hour_mlp_531),cmap=cmap_2_colors_rmse,vmin=0,vmax=400)
im_trans=ax[2].imshow(RMSE_array_hour_531,cmap=cmap_2_colors_rmse,vmin=0,vmax=400)

# set y axis plot 1 the time
ax[0].set_ylabel('Time of the day [hh:mm:ss]')
my_yticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']
ax[0].set_yticks(np.linspace(0,23,24))
ax[0].set_yticklabels(my_yticks,rotation=0,fontsize=9)

# remove y labels of plot 2 en 3
ax[1].set_yticklabels([0],'')#,fontsize=8)
ax[2].set_yticklabels([0],'')#,fontsize=8)

# remove x labels of plot 1 en 2
my_xticks_rf=['']
ax[0].set_xticks(np.array([0]))
ax[0].set_xticklabels(my_xticks_rf)#,fontsize=8)

my_xticks_mlp=['']
ax[1].set_xticks(np.array([0]))
ax[1].set_xticklabels(my_xticks_mlp)#,fontsize=8)

# set x labels of transformer plot
my_xticks_trans=['1','3','5','7','9','11','13','15','17','19','21','23']
ax[2].set_xticks(np.array([0,2,4,6,8,10,12,14,16,18,20,22]))
ax[2].set_xticklabels(my_xticks_trans)#,fontsize=8)
ax[2].set_xlabel('Prediction horizon [h]')#,fontsize=8)

#Set titles subplots
ax[0].set_title('RF',fontsize=10)
ax[1].set_title('MLP',fontsize=10)
ax[2].set_title('Transformer',fontsize=10)

# plot colorbar
cbar=fig.colorbar(im_trans)
cbar.ax.set_ylabel('RMSE [veh/h]', rotation=(360-90),fontsize=9,labelpad=10)

plt.subplots_adjust(left=0.1, bottom=0.1,  right=0.9, top=0.86,  wspace=0.1,hspace=0.1)

# %%Make delta figure for both locations
# Plot the rmse and mae over the prediction horizon for the multiple models
# Locatie 501 
pred_horizon=24
RMSE_array_test_501=[]
RMSE_array_test_531=[]
RMSE_array_test_531_removed=[]

for i in range(pred_horizon):
    y_pred_horizon_501=trans_y_pred_test_501[:,i]
    y_true_horizon_501=trans_y_true_test_501[:,i]

    y_pred_horizon_531=trans_y_pred_test_531[:,i]
    y_true_horizon_531=trans_y_true_test_531[:,i]

    y_pred_horizon_531_removed=np.delete(y_pred_horizon_531,slice(None,None,17))
    y_true_horizon_531_removed=np.delete(y_true_horizon_531,slice(None,None,17))
    
    
    RMSE_train_501=np.sqrt(mean_squared_error(y_pred_horizon_501,y_true_horizon_501))
    RMSE_train_531=np.sqrt(mean_squared_error(y_pred_horizon_531,y_true_horizon_531))
    RMSE_train_531_removed=np.sqrt(mean_squared_error(y_pred_horizon_531_removed,y_true_horizon_531_removed))
    
    
    RMSE_array_test_501.append(RMSE_train_501)
    RMSE_array_test_531.append(RMSE_train_531)
    RMSE_array_test_531_removed.append(RMSE_train_531_removed)
    


# Locatie 501 Figure
prediction_horizons=np.linspace(1,pred_horizon,pred_horizon)
RMSE_array_rf_501= np.full((24,),RMSE_501_test_rf)
RMSE_array_mlp_501= np.full((24,),RMSE_501_test_mlp)


plt.figure()
plt.plot(prediction_horizons,RMSE_array_test_501,label='Transformer',color=siemens_groen)
plt.plot(prediction_horizons,RMSE_array_rf_501,label='Random forest',color=siemens_groen,linestyle='--')
plt.plot(prediction_horizons,RMSE_array_mlp_501,label='Multilayer perceptron',color=siemens_groen,linestyle=':')

plt.legend(loc=1)
plt.xlabel('Prediction horizon [h]')
plt.ylabel('RMSE [veh/h]')
plt.title(' Performance for location 501')
plt.ylim((0,180))
plt.xlim((0.9,24))

# Locatie 531 Figure
RMSE_array_rf_531= np.full((24,),RMSE_531_test_rf)
RMSE_array_mlp_531= np.full((24,),RMSE_531_test_mlp)
# op h 3 eroverheen
point1=[2.8,2.8]
point2=[0,180]

plt.figure()
plt.plot(prediction_horizons,RMSE_array_test_531,label='Transformer',color=siemens_groen)
plt.plot(prediction_horizons,RMSE_array_rf_531,label='Random forest',color=siemens_groen,linestyle='--')
plt.plot(prediction_horizons,RMSE_array_mlp_531,label='Multilayer perceptron',color=siemens_groen,linestyle=':')
plt.plot(point1,point2, linestyle=':',color='k',linewidth=0.7)
plt.text(2.5,-10,'$x$')

plt.legend(loc=1)
plt.xlabel('Prediction horizon [h]')
plt.ylabel('RMSE [veh/h]')
plt.title('Performance for location 531')
plt.ylim((0,180))
plt.xlim((0.9,24))

#%% RELATIVE ERROR AND UNCERTAINTY

# take max of denumerator
rf_y_true_test_501_max=rf_y_true_test_501
rf_y_true_test_501_max[rf_y_true_test_501<1]=1
mlp_y_true_test_501_max=mlp_y_true_test_501
mlp_y_true_test_501_max[mlp_y_true_test_501<1]=1
trans_y_true_test_501_max=trans_y_true_test_501
trans_y_true_test_501_max[trans_y_true_test_501<1]=1

rf_y_true_test_531_max=rf_y_true_test_531
rf_y_true_test_531_max[rf_y_true_test_531<1]=1
mlp_y_true_test_531_max=mlp_y_true_test_531
mlp_y_true_test_531_max[mlp_y_true_test_531<1]=1
trans_y_true_test_531_max=trans_y_true_test_531
trans_y_true_test_531_max[trans_y_true_test_531<1]=1

# Make sure mlp does not make negative predictions
mlp_y_pred_test_501[mlp_y_pred_test_501<1]=1
mlp_y_pred_test_531[mlp_y_pred_test_531<1]=1

#calculate the error
rel_error_501_rf=(((rf_y_pred_test_501-rf_y_true_test_501)/(rf_y_true_test_501_max))*100)
rel_error_501_mlp=(((mlp_y_pred_test_501-mlp_y_true_test_501)/(mlp_y_true_test_501_max))*100)
rel_error_501_trans=(((trans_y_pred_test_501-trans_y_true_test_501)/(trans_y_true_test_501_max))*100)

rel_error_531_rf=(((rf_y_pred_test_531-rf_y_true_test_531)/(rf_y_true_test_531_max))*100)
rel_error_531_mlp=(((mlp_y_pred_test_531-mlp_y_true_test_531)/(mlp_y_true_test_531_max))*100)
rel_error_531_trans=(((trans_y_pred_test_531-trans_y_true_test_531)/(trans_y_true_test_531_max))*100)

# make transformer errors flat
rel_error_501_trans_flat=np.concatenate(rel_error_501_trans)
rel_error_531_trans_flat=np.concatenate(rel_error_531_trans)


# %%Step 2 investigate the distribution of the errors
#location 501
def investigate_error_distribution(rel_error,title):
    freq,bins=np.histogram(rel_error,bins=19000,range=[rel_error.min(),rel_error.max()])
    freq_threshold=freq.astype('float32')
    
    histogram_df=pd.DataFrame({'freq_threshold':freq_threshold,'bins':bins[:-1]})
    histogram_df.loc[histogram_df.index[histogram_df['freq_threshold']<1],'freq_threshold']=np.nan
    
    histogram_df_na=histogram_df.copy(deep=True)
    histogram_df_na=histogram_df_na.dropna()
  
    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    #plt.tight_layout()
    axs[0].plot(histogram_df_na['bins'],histogram_df_na['freq_threshold'],'-',linewidth=1,color=siemens_groen,label='frequency')   
    axs[1].boxplot(rel_error,vert=0,widths=[0.75])
    axs[1].set(xlabel='Relative error [%]',ylabel='[-]')
    axs[0].set(ylabel='Frequency')


investigate_error_distribution(rel_error_501_rf,'Distribution error location 501 random forest')
investigate_error_distribution(rel_error_501_mlp,'Distribution error location 501 multilayer perceptron')
investigate_error_distribution(rel_error_501_trans_flat,'Distribution error location 501 transformer')

investigate_error_distribution(rel_error_531_rf,'Distribution error location 531 random forest')
investigate_error_distribution(rel_error_531_mlp,'Distribution error location 531 multilayer perceptron')
investigate_error_distribution(rel_error_531_trans_flat,'Distribution error location 531 transformer')

#%% Investigate distribution transformer error certain start time and prediction horizon
def investigate_error_horizon_time(rel_error,title,time,horizon):
    error_hour=rel_error[time::24,:]
    yshape=int(rel_error.shape[0]/24)
    error_hour_horizon=error_hour[:,horizon].reshape(yshape,1)
    
    freq,bins=np.histogram(error_hour_horizon,bins=800,range=[error_hour_horizon.min(),error_hour_horizon.max()])
    freq_threshold=freq.astype('float32')
    
    histogram_df=pd.DataFrame({'freq_threshold':freq_threshold,'bins':bins[:-1]})
    histogram_df.loc[histogram_df.index[histogram_df['freq_threshold']<1],'freq_threshold']=np.nan
    
    histogram_df_na=histogram_df.copy(deep=True)
    histogram_df_na=histogram_df_na.dropna()

    fig, axs = plt.subplots(2)
    fig.suptitle(title)
    #plt.tight_layout()
    axs[0].plot(histogram_df_na['bins'],histogram_df_na['freq_threshold'],'-',linewidth=1,color=siemens_groen,label='frequency')   
    axs[1].boxplot(error_hour_horizon,vert=0,widths=[0.75])
    axs[1].set(xlabel='Relative error [%]',ylabel='[-]')
    axs[0].set(ylabel='Frequency')

    
t=0
h=23
for h in range(24):
    investigate_error_horizon_time(rel_error_501_trans,'Distribution error location 501 transformer time {} horizon {}'.format(t,h),t,h)
    investigate_error_horizon_time(rel_error_531_trans,'Distribution error location 531 transformer time {} horizon {}'.format(t,h),t,h)

#%% Waar is de hoogste error?
print('rf 501',np.where(rel_error_501_rf == np.amax(rel_error_501_rf))) 
print('mlp 501',np.where(rel_error_501_mlp == np.amax(rel_error_501_mlp))) 
print('trans 501',np.where(rel_error_501_trans == np.amax(rel_error_501_trans))) # 3595, start op 19 over en array 13
print(10*'*')
print('rf 531',np.where(rel_error_531_rf == np.amax(rel_error_531_rf))) 
print('mlp 531',np.where(rel_error_531_mlp == np.amax(rel_error_531_mlp))) 
print('trans 531',np.where(rel_error_531_trans == np.amax(rel_error_531_trans)))  # 3542, 24 ovr en start op 18
print(10*'*')

#%%
# investigate for locatie 501
xticks_4juni = ['19:00:00','20:00:00','21:00:00','22:00:00','23:00:00','00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00']
start=3595

predictions=np.linspace(48,72,24)
fig,ax =plt.subplots()
# plot historical traffic flow
plt.plot(loc_501.loc[start+17016:start+17016+72,'waarnemingen_intensiteit'].reset_index(drop=True),label='True',color=o1)
# plot predictions
plt.plot(predictions,trans_y_pred_test_501[start],label='Transformer ',color=siemens_groen)
plt.plot(predictions,rf_y_pred_test_501[start+48:start+48+24],label='Random forest ',color=siemens_groen,linestyle='--')
plt.plot(predictions,mlp_y_pred_test_501[start+48:start+48+24],label='Multilayer perceptron ',color=siemens_groen,linestyle=':')

# plot line where start t
point1=[47,47]
point2=[0,1400]
plt.plot(point1,point2, linestyle='--',color='k',linewidth=0.7)


plt.title('Unexpected traffic flow for location 501')
plt.legend(loc=2)
x_ticks_date=['06-03-2019','06-04-2019','06-05-2019']

loc_ticks=np.array([19,19+24,19+2*24])
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_date)
plt.xticks(rotation=0)

plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1400))
plt.xlim((0,72))
ax.fill_between([0,47], [1400,1400],color=o1,alpha=0.1)

start=3595+13

predictions=np.linspace(48,72,24)
fig,ax =plt.subplots()
#plt.plot(predictions,trans_y_true_test_501[start],label='True',color=siemens_blauw)
# plot historical traffic flow
plt.plot(loc_501.loc[start+17016:start+17016+72,'waarnemingen_intensiteit'].reset_index(drop=True),label='True',color=o1)
# plot predictions
plt.plot(predictions,trans_y_pred_test_501[start],label='Transformer ',color=siemens_groen)
plt.plot(predictions,rf_y_pred_test_501[start+48:start+48+24],label='Random forest ',color=siemens_groen,linestyle='--')
plt.plot(predictions,mlp_y_pred_test_501[start+48:start+48+24],label='Multilayer perceptron ',color=siemens_groen,linestyle=':')

# plot line where start t
point1=[47,47]
point2=[0,1400]
plt.plot(point1,point2, linestyle='--',color='k',linewidth=0.7)


plt.title('Unexpected traffic flow for location 501')
plt.legend(loc=2)
x_ticks_date=['06-03-2019','06-04-2019','06-05-2019']

loc_ticks=np.array([19,19+24,19+2*24])-13
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_date)
plt.xticks(rotation=0)

plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1400))
ax.fill_between([0,47], [1400,1400],color=o1,alpha=0.1)
plt.xlim((0,73))


#%% 4/5mei
# start index 2019-05-04 19896
predictions=np.linspace(48,72,24)
fig,ax =plt.subplots()
#plt.plot(predictions,trans_y_true_test_501[start],label='True',color=siemens_blauw)
# plot historical traffic flow
plt.plot(loc_501.loc[19896-49:19896+23,'waarnemingen_intensiteit'].reset_index(drop=True),color=o1,label='True')
# plot predictions
plt.plot(predictions,trans_y_pred_test_501[19896-17016-48,:],label='Transformer ',color=siemens_groen)
plt.plot(predictions,rf_y_pred_test_501[19896-17016:19896-17016+24],label='Random forest ',color=siemens_groen,linestyle='--')
plt.plot(predictions,mlp_y_pred_test_501[19896-17016:19896-17016+24],label='Multilayer perceptron ',color=siemens_groen,linestyle=':')

# plot line where start t
point1=[47,47]
point2=[0,1400]
plt.plot(point1,point2, linestyle='--',color='k',linewidth=0.7)


plt.title('Unexpected traffic flow for location 501')
plt.legend(loc=2)
x_ticks_date=['05-02-2019','05-03-2019','05-04-2019']

loc_ticks=np.array([12,12+24,12+2*24])
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_date)
plt.xticks(rotation=0)

plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1400))
ax.fill_between([0,47], [1400,1400],color=o1,alpha=0.1)
plt.xlim((0,73))

#%% Uncertainty, median etc distribution of the errors

median_trans_501=np.zeros((24,24))
percentile_25_trans_501=np.zeros((24,24))
percentile_75_trans_501=np.zeros((24,24))
percentile_5_trans_501=np.zeros((24,24))
percentile_95_trans_501=np.zeros((24,24))

pred_horizon=24
for h in range(pred_horizon): # for each start time
    error_rel_hour=rel_error_501_trans[h::pred_horizon,:]
    for i in range(pred_horizon):
        error_rel_horizon=error_rel_hour[:,i].reshape(353,1)
        
        median_1 = np.percentile(error_rel_horizon,50)
        q_25 =np.percentile(error_rel_horizon,25)
        q_75 = np.percentile(error_rel_horizon,75)
        q_5= np.percentile(error_rel_horizon,5)
        q_95= np.percentile(error_rel_horizon,95)
              
        median_1_around=1/(1+median_1/100)
        q_25_around=1/(1+q_25/100)
        q_75_around=1/(1+q_75/100)
        q_5_around=1/(1+q_5/100)
        q_95_around=1/(1+q_95/100)
         
        median_trans_501[h,i]=median_1_around
        percentile_25_trans_501[h,i]=q_25_around
        percentile_75_trans_501[h,i]=q_75_around
        percentile_5_trans_501[h,i]=q_5_around
        percentile_95_trans_501[h,i]=q_95_around
           
vmin_set=0
vmax_set=3
cmap_weights=LinearSegmentedColormap.from_list("",[o1,wit,siemens_groen_light1,siemens_groen])

# plot differently
median_trans_501_dif=np.zeros((24,24))
percentile_25_trans_501_dif=np.zeros((24,24))
percentile_75_trans_501_dif=np.zeros((24,24))
percentile_5_trans_501_dif=np.zeros((24,24))
percentile_95_trans_501_dif=np.zeros((24,24))

median_trans_531_dif=np.zeros((24,24))
percentile_25_trans_531_dif=np.zeros((24,24))
percentile_75_trans_531_dif=np.zeros((24,24))
percentile_5_trans_531_dif=np.zeros((24,24))
percentile_95_trans_531_dif=np.zeros((24,24))


pred_horizon=24
for h in range(pred_horizon): # for each start time
    error_rel_hour=rel_error_501_trans[:,h]
    for i in range(pred_horizon):
        error_rel_horizon=error_rel_hour[i::24]
        
        median_1 = np.percentile(error_rel_horizon,50)
        q_25 =np.percentile(error_rel_horizon,25)
        q_75 = np.percentile(error_rel_horizon,75)
        q_5= np.percentile(error_rel_horizon,5)
        q_95= np.percentile(error_rel_horizon,95)
              
        median_1_around=1/(1+median_1/100)
        q_25_around=1/(1+q_25/100)
        q_75_around=1/(1+q_75/100)
        q_5_around=1/(1+q_5/100)
        q_95_around=1/(1+q_95/100)
        
        a=i+h
        if a>23:
            a_final=a-24
        else:
            a_final=a
            
            
        median_trans_501_dif[a_final,h]=median_1_around
        percentile_25_trans_501_dif[a_final,h]=q_25_around
        percentile_75_trans_501_dif[a_final,h]=q_75_around
        percentile_5_trans_501_dif[a_final,h]=q_5_around
        percentile_95_trans_501_dif[a_final,h]=q_95_around
        
for h in range(pred_horizon): # for each start time
    error_rel_hour=rel_error_531_trans[:,h]
    for i in range(pred_horizon):
        error_rel_horizon=error_rel_hour[i::24]
        
        median_1 = np.percentile(error_rel_horizon,50)
        q_25 =np.percentile(error_rel_horizon,25)
        q_75 = np.percentile(error_rel_horizon,75)
        q_5= np.percentile(error_rel_horizon,5)
        q_95= np.percentile(error_rel_horizon,95)
              
        median_1_around=1/(1+median_1/100)
        q_25_around=1/(1+q_25/100)
        q_75_around=1/(1+q_75/100)
        q_5_around=1/(1+q_5/100)
        q_95_around=1/(1+q_95/100)
        
        a=i+h
        if a>23:
            a_final=a-24
        else:
            a_final=a
            
            
        median_trans_531_dif[a_final,h]=median_1_around
        percentile_25_trans_531_dif[a_final,h]=q_25_around
        percentile_75_trans_531_dif[a_final,h]=q_75_around
        percentile_5_trans_531_dif[a_final,h]=q_5_around
        percentile_95_trans_531_dif[a_final,h]=q_95_around
    


#%% plot heatmpa of the uncertainty ranges
def plot_alpha(alpha,title):
    plt.figure()     
    plt.imshow(alpha,cmap=cmap_weights,vmin=vmin_set,vmax=vmax_set)
    plt.colorbar()
    plt.xlabel('prediction horizon [h]')
    plt.ylabel('start time [hour of the day]')
    plt.title(title)
    
plot_alpha(percentile_5_trans_501_dif,r' $\alpha_{t,h,5}$ location 501')
plot_alpha(percentile_25_trans_501_dif,r' $\alpha_{t,h,25}$ location 501')
plot_alpha(median_trans_501_dif,r' $\alpha_{t,h,50}$ location 501')
plot_alpha(percentile_75_trans_501_dif,r' $\alpha_{t,h,75}$ location 501')
plot_alpha(percentile_95_trans_501_dif,r' $\alpha_{t,h,95}$ location 501')

# plot for random forest and mlp
median_rf_501=np.zeros((24,1))
percentile_25_rf_501=np.zeros((24,1))
percentile_75_rf_501=np.zeros((24,1))
percentile_5_rf_501=np.zeros((24,1))
percentile_95_rf_501=np.zeros((24,1))

median_mlp_501=np.zeros((24,1))
percentile_25_mlp_501=np.zeros((24,1))
percentile_75_mlp_501=np.zeros((24,1))
percentile_5_mlp_501=np.zeros((24,1))
percentile_95_mlp_501=np.zeros((24,1))

median_rf_531=np.zeros((24,1))
percentile_25_rf_531=np.zeros((24,1))
percentile_75_rf_531=np.zeros((24,1))
percentile_5_rf_531=np.zeros((24,1))
percentile_95_rf_531=np.zeros((24,1))

median_mlp_531=np.zeros((24,1))
percentile_25_mlp_531=np.zeros((24,1))
percentile_75_mlp_531=np.zeros((24,1))
percentile_5_mlp_531=np.zeros((24,1))
percentile_95_mlp_531=np.zeros((24,1))

for h in range(24): # for each start time
    
    error_rel_horizon=rel_error_501_rf[h::24,:]

    median_1 = np.percentile(error_rel_horizon,50)
    q_25 =np.percentile(error_rel_horizon,25)
    q_75 = np.percentile(error_rel_horizon,75)
    q_5= np.percentile(error_rel_horizon,5)
    q_95= np.percentile(error_rel_horizon,95)
              
    median_1_around=1/(1+median_1/100)
    q_25_around=1/(1+q_25/100)
    q_75_around=1/(1+q_75/100)
    q_5_around=1/(1+q_5/100)
    q_95_around=1/(1+q_95/100)
         
    median_rf_501[h,0]=median_1_around
    percentile_25_rf_501[h,0]=q_25_around
    percentile_75_rf_501[h,0]=q_75_around
    percentile_5_rf_501[h,0]=q_5_around
    percentile_95_rf_501[h,0]=q_95_around
    
    error_rel_horizon=rel_error_501_mlp[h::24,:]    
    median_1 = np.percentile(error_rel_horizon,50)
    q_25 =np.percentile(error_rel_horizon,25)
    q_75 = np.percentile(error_rel_horizon,75)
    q_5= np.percentile(error_rel_horizon,5)
    q_95= np.percentile(error_rel_horizon,95)
    #print(q_5) # altijd groter dan -100
    median_1_around=1/(1+median_1/100)
    q_25_around=1/(1+q_25/100)
    q_75_around=1/(1+q_75/100)
    q_5_around=1/(1+q_5/100)
    q_95_around=1/(1+q_95/100)
    print(q_5_around)
    median_mlp_501[h,0]=median_1_around
    percentile_25_mlp_501[h,0]=q_25_around
    percentile_75_mlp_501[h,0]=q_75_around
    percentile_5_mlp_501[h,0]=q_5_around
    percentile_95_mlp_501[h,0]=q_95_around
    
    error_rel_horizon=rel_error_531_rf[h::24,:]

    median_1 = np.percentile(error_rel_horizon,50)
    q_25 =np.percentile(error_rel_horizon,25)
    q_75 = np.percentile(error_rel_horizon,75)
    q_5= np.percentile(error_rel_horizon,5)
    q_95= np.percentile(error_rel_horizon,95)
              
    median_1_around=1/(1+median_1/100)
    q_25_around=1/(1+q_25/100)
    q_75_around=1/(1+q_75/100)
    q_5_around=1/(1+q_5/100)
    q_95_around=1/(1+q_95/100)
         
    median_rf_531[h,0]=median_1_around
    percentile_25_rf_531[h,0]=q_25_around
    percentile_75_rf_531[h,0]=q_75_around
    percentile_5_rf_531[h,0]=q_5_around
    percentile_95_rf_531[h,0]=q_95_around
    
    error_rel_horizon=rel_error_531_mlp[h::24,:]

    median_1 = np.percentile(error_rel_horizon,50)
    q_25 =np.percentile(error_rel_horizon,25)
    q_75 = np.percentile(error_rel_horizon,75)
    q_5= np.percentile(error_rel_horizon,5)
    q_95= np.percentile(error_rel_horizon,95)
    print('h',h)
    median_1_around=1/(1+median_1/100)
    q_25_around=1/(1+q_25/100)
    q_75_around=1/(1+q_75/100)
    q_5_around=1/(1+q_5/100)
    q_95_around=1/(1+q_95/100)
    print('q5 around',q_5_around)
         
    median_mlp_531[h,0]=median_1_around
    percentile_25_mlp_531[h,0]=q_25_around
    percentile_75_mlp_531[h,0]=q_75_around
    percentile_5_mlp_531[h,0]=q_5_around
    percentile_95_mlp_531[h,0]=q_95_around
    
plot_alpha(percentile_5_rf_501,r' $\alpha_{t,h,5}$ location 501')
plot_alpha(percentile_25_rf_501,r' $\alpha_{t,h,25}$ location 501')
plot_alpha(median_rf_501,r' $\alpha_{t,h,50}$ location 501')
plot_alpha(percentile_75_rf_501,r' $\alpha_{t,h,75}$ location 501')
plot_alpha(percentile_95_rf_501,r' $\alpha_{t,h,95}$ location 501')

plot_alpha(percentile_5_mlp_501,r' $\alpha_{t,h,5}$ location 501')
plot_alpha(percentile_25_mlp_501,r' $\alpha_{t,h,25}$ location 501')
plot_alpha(median_mlp_501,r' $\alpha_{t,h,50}$ location 501')
plot_alpha(percentile_75_mlp_501,r' $\alpha_{t,h,75}$ location 501')
plot_alpha(percentile_95_mlp_501,r' $\alpha_{t,h,95}$ location 501')
 

#%% Everything in one figure 
# LOCATION 501
alpha_rf_501=[percentile_5_rf_501,percentile_25_rf_501,median_rf_501,percentile_75_rf_501,percentile_95_rf_501]
alpha_mlp_501=[percentile_5_mlp_501,percentile_25_mlp_501,median_mlp_501,percentile_75_mlp_501,percentile_95_mlp_501]
alpha_trans_501=[percentile_5_trans_501_dif,percentile_25_trans_501_dif,median_trans_501_dif,percentile_75_trans_501_dif,percentile_95_trans_501_dif]

name=[r' $\alpha_{t,h,5}$',r' $\alpha_{t,h,25}$',r' $\alpha_{t,h,50}$',r' $\alpha_{t,h,75}$',r' $\alpha_{t,h,95}$']
cmap_weights=LinearSegmentedColormap.from_list("",[o1,wit,siemens_groen_light3,siemens_groen_light1,siemens_groen])
cmap_weights_mlp=LinearSegmentedColormap.from_list("",[o1,wit,siemens_groen_light3,siemens_groen_light1,siemens_groen,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw])
savenames_501=['501_5','501_25','501_50','501_75','501_95']
for i in range(5):
    fig,ax=plt.subplots(nrows=1,ncols=3, gridspec_kw={'width_ratios': [ 1,1,24]})
    fig.suptitle('{} location 501'.format(name[i]), fontsize=12)
    if i==0:
        im_mlp=ax[1].imshow(alpha_mlp_501[i],cmap=cmap_weights_mlp,vmin=0,vmax=20)
    else:
        im_mlp=ax[1].imshow(alpha_mlp_501[i],cmap=cmap_weights,vmin=0,vmax=4)
    im_rf=ax[0].imshow(alpha_rf_501[i],cmap=cmap_weights,vmin=0,vmax=4)
    im_trans=ax[2].imshow(alpha_trans_501[i],cmap=cmap_weights,vmin=0,vmax=4)
    
    # set y axis plot 1 the time
    ax[0].set_ylabel('Time of the day [hh:mm:ss]')
    my_yticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']
    ax[0].set_yticks(np.linspace(0,23,24))
    ax[0].set_yticklabels(my_yticks,rotation=0,fontsize=9)
    
    # remove y labels of plot 2 en 3
    ax[1].set_yticklabels([0],'')#,fontsize=8)
    ax[2].set_yticklabels([0],'')#,fontsize=8)
    
    # remove x labels of plot 1 en 2
    my_xticks_rf=['']
    ax[0].set_xticks(np.array([0]))
    ax[0].set_xticklabels(my_xticks_rf)#,fontsize=8)
    
    my_xticks_mlp=['']
    ax[1].set_xticks(np.array([0]))
    ax[1].set_xticklabels(my_xticks_mlp)#,fontsize=8)
    
    # set x labels of transformer plot
    my_xticks_trans=['1','3','5','7','9','11','13','15','17','19','21','23']
    ax[2].set_xticks(np.array([0,2,4,6,8,10,12,14,16,18,20,22]))
    ax[2].set_xticklabels(my_xticks_trans)#,fontsize=8)
    ax[2].set_xlabel('Prediction horizon [h]')#,fontsize=8)
    
    #Set titles subplots
    ax[0].set_title('RF',fontsize=10)
    ax[1].set_title('MLP',fontsize=10)
    ax[2].set_title('Transformer',fontsize=10)
    # plot colorbar
    cbar=fig.colorbar(im_trans)
    
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.86,wspace=0.1,hspace=0.1)

# %%LOCATION 531
alpha_rf_531=[percentile_5_rf_531,percentile_25_rf_531,median_rf_531,percentile_75_rf_531,percentile_95_rf_531]
alpha_mlp_531=[percentile_5_mlp_531,percentile_25_mlp_531,median_mlp_531,percentile_75_mlp_531,percentile_95_mlp_531]
alpha_trans_531=[percentile_5_trans_531_dif,percentile_25_trans_531_dif,median_trans_531_dif,percentile_75_trans_531_dif,percentile_95_trans_531_dif]

name=[r' $\alpha_{t,h,5}$',r' $\alpha_{t,h,25}$',r' $\alpha_{t,h,50}$',r' $\alpha_{t,h,75}$',r' $\alpha_{t,h,95}$']
cmap_weights=LinearSegmentedColormap.from_list("",[o1,wit,siemens_groen_light3,siemens_groen_light1,siemens_groen])
cmap_weights_mlp=LinearSegmentedColormap.from_list("",[o1,wit,siemens_groen_light3,siemens_groen_light1,siemens_groen,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw,siemens_blauw])

savenames_531=['531_5','531_25','531_50','531_75','531_95']

for i in range(5):
    fig,ax=plt.subplots(nrows=1,ncols=3, gridspec_kw={'width_ratios': [ 1,1,24]})
    fig.suptitle('{} location 531'.format(name[i]), fontsize=12)
    if i<2:
        im_mlp=ax[1].imshow(alpha_mlp_531[i],cmap=cmap_weights_mlp,vmin=0,vmax=32)
    else:
        im_mlp=ax[1].imshow(alpha_mlp_531[i],cmap=cmap_weights,vmin=0,vmax=4)

    im_rf=ax[0].imshow(alpha_rf_531[i],cmap=cmap_weights,vmin=0,vmax=4)
    im_trans=ax[2].imshow(alpha_trans_531[i],cmap=cmap_weights,vmin=0,vmax=4)
    
    # set y axis plot 1 the time
    ax[0].set_ylabel('Time of the day [hh:mm:ss]')
    my_yticks = ['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00','05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','10:00:00','11:00:00','12:00:00','13:00:00','14:00:00','15:00:00','16:00:00','17:00:00','18:00:00','19:00:00','20:00:00','21:00:00','22:00:00','23:00:00']
    ax[0].set_yticks(np.linspace(0,23,24))
    ax[0].set_yticklabels(my_yticks,rotation=0,fontsize=9)
    
    # remove y labels of plot 2 en 3
    ax[1].set_yticklabels([0],'')#,fontsize=8)
    ax[2].set_yticklabels([0],'')#,fontsize=8)
    
    # remove x labels of plot 1 en 2
    my_xticks_rf=['']
    ax[0].set_xticks(np.array([0]))
    ax[0].set_xticklabels(my_xticks_rf)#,fontsize=8)
    
    my_xticks_mlp=['']
    ax[1].set_xticks(np.array([0]))
    ax[1].set_xticklabels(my_xticks_mlp)#,fontsize=8)
    
    # set x labels of transformer plot
    my_xticks_trans=['1','3','5','7','9','11','13','15','17','19','21','23']
    ax[2].set_xticks(np.array([0,2,4,6,8,10,12,14,16,18,20,22]))
    ax[2].set_xticklabels(my_xticks_trans)#,fontsize=8)
    ax[2].set_xlabel('Prediction horizon [h]')#,fontsize=8)
    
    #Set titles subplots
    ax[0].set_title('RF',fontsize=10)
    ax[1].set_title('MLP',fontsize=10)
    ax[2].set_title('Transformer',fontsize=10)
    # plot colorbar
    cbar=fig.colorbar(im_trans)
    
    plt.subplots_adjust(left=0.1,bottom=0.1,right=0.9,top=0.86,wspace=0.1,hspace=0.1)



#%% Plot the errors made throughout the year 
# 1. Change back the order of the train and validation set, such that the i know which day which error is made.
def change_back_data_train_val(data_train_true,data_val_true,data_train_pred,data_val_pred):
    size=data_train_true.shape[0]+data_val_true.shape[0]
    
    train_val_true=np.concatenate((data_train_true,data_val_true)).reshape(size,)
    train_val_pred=np.concatenate((data_train_pred,data_val_pred)).reshape(size,)
    
    np.random.seed(42)
    shuffled_indices=np.random.permutation(size)
    
    y_df=pd.DataFrame({'original_index':shuffled_indices,'true':train_val_true,'pred':train_val_pred})
    y_df_back=y_df.sort_values(by='original_index').reset_index(drop=True)
    del y_df_back['original_index']
    
    return y_df_back

y_df_back_501_rf=change_back_data_train_val(rf_y_true_train_501,rf_y_true_val_501,rf_y_pred_train_501,rf_y_pred_val_501)
y_df_back_501_mlp=change_back_data_train_val(mlp_y_true_train_501,mlp_y_true_val_501,mlp_y_pred_train_501,mlp_y_pred_val_501)
y_df_back_531_rf=change_back_data_train_val(rf_y_true_train_531,rf_y_true_val_531,rf_y_pred_train_531,rf_y_pred_val_531)
y_df_back_531_mlp=change_back_data_train_val(mlp_y_true_train_531,mlp_y_true_val_531,mlp_y_pred_train_531,mlp_y_pred_val_531)

#concatenate test set aswell
y_df_test_501_rf=pd.DataFrame({'true':rf_y_true_test_501.reshape(rf_y_true_test_501.shape[0],),'pred':rf_y_pred_test_501.reshape(rf_y_true_test_501.shape[0],)})
y_df_total_501_rf=pd.concat([y_df_back_501_rf,y_df_test_501_rf],axis=0).reset_index(drop=True)

y_df_test_501_mlp=pd.DataFrame({'true':mlp_y_true_test_501.reshape(mlp_y_true_test_501.shape[0],),'pred':mlp_y_pred_test_501.reshape(mlp_y_true_test_501.shape[0],)})
y_df_total_501_mlp=pd.concat([y_df_back_501_mlp,y_df_test_501_mlp],axis=0).reset_index(drop=True)

y_df_test_531_rf=pd.DataFrame({'true':rf_y_true_test_531.reshape(rf_y_true_test_531.shape[0],),'pred':rf_y_pred_test_531.reshape(rf_y_true_test_531.shape[0],)})
y_df_total_531_rf=pd.concat([y_df_back_531_rf,y_df_test_531_rf],axis=0).reset_index(drop=True)

y_df_test_531_mlp=pd.DataFrame({'true':mlp_y_true_test_531.reshape(mlp_y_true_test_531.shape[0],),'pred':mlp_y_pred_test_531.reshape(mlp_y_true_test_531.shape[0],)})
y_df_total_531_mlp=pd.concat([y_df_back_531_mlp,y_df_test_531_mlp],axis=0).reset_index(drop=True)

# extra no march
y_df_back_531_rf_nomarch=change_back_data_train_val(rf_y_true_train_531_nomarch,rf_y_true_val_531_nomarch,rf_y_pred_train_531_nomarch,rf_y_pred_val_531_nomarch)
y_df_back_531_mlp_nomarch=change_back_data_train_val(mlp_y_true_train_531_nomarch,mlp_y_true_val_531_nomarch,mlp_y_pred_train_531_nomarch,mlp_y_pred_val_531_nomarch)

y_df_test_531_mlp_nomarch=pd.DataFrame({'true':mlp_y_true_test_531_nomarch.reshape(mlp_y_true_test_531_nomarch.shape[0],),'pred':mlp_y_pred_test_531_nomarch.reshape(mlp_y_true_test_531_nomarch.shape[0],)})
y_df_total_531_mlp_nomarch=pd.concat([y_df_back_531_mlp_nomarch,y_df_test_531_mlp_nomarch],axis=0).reset_index(drop=True)

y_df_test_531_rf_nomarch=pd.DataFrame({'true':rf_y_true_test_531_nomarch.reshape(rf_y_true_test_531_nomarch.shape[0],),'pred':rf_y_pred_test_531_nomarch.reshape(rf_y_true_test_531_nomarch.shape[0],)})
y_df_total_531_rf_nomarch=pd.concat([y_df_back_531_rf_nomarch,y_df_test_531_rf_nomarch],axis=0).reset_index(drop=True)

#%%

def plot_error_calplot(data,loc_data,model,loc):
    data_error=data.copy()
    # maak nieuwe kolom met squared error
    data_error['error']=(data['true']-data['pred'])**2
    data_error['mae']=(np.abs(data['true']-data['pred']))
   
    # add start_datum and start_tijd as a columns
    data_error['start_datum']=pd.to_datetime(loc_data['start_datum'][0:data.shape[0]]).reset_index(drop=True)
    data_error['start_tijd']=pd.to_datetime(loc_data['start_tijd'][0:data.shape[0]]).reset_index(drop=True)  
    
    print(data_error['start_datum'])
    RMSE_pred=data_error.pivot_table(columns='start_tijd',values='error',index='start_datum')
    MAE_pred=data_error.pivot_table(columns='start_tijd',values='mae',index='start_datum')

    column_list=list(RMSE_pred)

    RMSE_pred["daily_rmse"] = np.sqrt(RMSE_pred[column_list].sum(axis=1)/24)
    MAE_pred["daily_mae"] = MAE_pred[column_list].sum(axis=1)/24
    
    # rmse gemiddeld over de dag
    # change vmax to 500 if train
    cmap_rmse = LinearSegmentedColormap.from_list("", [wit,o1])
    cmap_mae = LinearSegmentedColormap.from_list("", [wit,o1])
    
    
    fontproperties = fm.FontProperties(size=18)
    suptitle_kws = dict(fontproperties=fontproperties)
    
    calplot.calplot(RMSE_pred['daily_rmse'],cmap=cmap_rmse,textfiller='-',colorbar=True,vmax=530,suptitle='RMSE for the {} at location {}'.format(model,loc),suptitle_kws=suptitle_kws)



plot_error_calplot(y_df_total_501_rf,loc_501,'random forest','501')
plot_error_calplot(y_df_total_501_mlp,loc_501,'multilayer perceptron','501')
plot_error_calplot(y_df_total_531_rf,loc_531,'random forest','531')
plot_error_calplot(y_df_total_531_mlp,loc_531,'multilayer perceptron','531')
loc_531_nomarch=loc_531.loc[((loc_531['start_datum']<'2017-03-01') | (loc_531['start_datum']>'2017-03-31')),:]


#%% For the transformer 
def change_back_data_train_val_trans(y_train_pred,y_train_true,y_val_pred,y_val_true):
    size=y_train_pred.shape[0]+y_val_pred.shape[0]
    
    train_val_true=np.concatenate((y_train_true,y_val_true)).reshape(size,24)
    train_val_pred=np.concatenate((y_train_pred,y_val_pred)).reshape(size,24)
    
    np.random.seed(42)
    shuffled_indices=np.random.permutation(size)
    y_df_true=pd.DataFrame(data=train_val_true,index=shuffled_indices,columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
    y_df_pred=pd.DataFrame(data=train_val_pred,index=shuffled_indices,columns=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24'])
   
    
    
    y_df_true_back=y_df_true.sort_index()
    y_df_pred_back=y_df_pred.sort_index()
    
    y_df_true_back_start=y_df_true_back.iloc[::24,:].reset_index(drop=True)
    y_df_pred_back_start=y_df_pred_back.iloc[::24,:].reset_index(drop=True)

    return y_df_true_back_start,y_df_pred_back_start

[trans_501_true_back,trans_501_pred_back]=change_back_data_train_val_trans(trans_y_pred_train_501,trans_y_true_train_501,trans_y_pred_val_501,trans_y_true_val_501)
[trans_531_true_back,trans_531_pred_back]=change_back_data_train_val_trans(trans_y_pred_train_531,trans_y_true_train_531,trans_y_pred_val_531,trans_y_true_val_531)
[trans_531_true_back_nomarch,trans_531_pred_back_nomarch]=change_back_data_train_val_trans(trans_y_pred_train_531_nomarch,trans_y_true_train_531_nomarch,trans_y_pred_val_531_nomarch,trans_y_true_val_531_nomarch)


def plot_error_calplot_train_test(y_pred_train,y_true_train,datum_train,y_pred_test,y_true_test,datum_test,title,loc):    
    
    y_pred_usefull=y_pred_train
    y_true_usefull=y_true_train
    fontproperties = fm.FontProperties(size=18)
    suptitle_kws = dict(fontproperties=fontproperties)
    
    # Maak nieuwe dataframe
    Error_df=(y_pred_usefull-y_true_usefull)**2    
    daily_rmse=np.sqrt(Error_df.sum(axis=1)/24)
    daily_rmse_df_train=pd.DataFrame({'rmse':daily_rmse})
    daily_rmse_df_train=daily_rmse_df_train.set_index(datum_train,drop=True)

    y_pred_array=pd.DataFrame(y_pred_test)
    y_true_array=pd.DataFrame(y_true_test)
    y_pred_usefull=y_pred_array.loc[(y_pred_array.index) % 24 ==0,:].reset_index(drop=True)
    y_true_usefull=y_true_array.loc[(y_true_array.index) % 24 ==0,:].reset_index(drop=True)
    
    Error_df=(y_pred_usefull-y_true_usefull)**2    
    daily_rmse=np.sqrt(Error_df.sum(axis=1)/24)
    daily_rmse_df_test=pd.DataFrame({'rmse':daily_rmse})
    daily_rmse_df_test=daily_rmse_df_test.set_index(datum_test,drop=True)
    daily_rmse_df_total=pd.concat([daily_rmse_df_train,daily_rmse_df_test])

    # rmse average over the day
    cmap= LinearSegmentedColormap.from_list("", [wit,o7,o1])
    calplot.calplot(daily_rmse_df_total['rmse'],cmap=cmap,textfiller='-',colorbar=True,suptitle='{}'.format(title),vmin=0,vmax=530,suptitle_kws=suptitle_kws)


#%% Look into location 531, March
loc_531_march_2017=loc_531.loc[((loc_531['start_datum']<'2017-04-01') & (loc_531['start_datum']>'2017-02-28')),:]
loc_531_march_2018=loc_531.loc[((loc_531['start_datum']<'2018-04-01') & (loc_531['start_datum']>'2018-02-28')),:]
loc_531_march_2019=loc_531.loc[((loc_531['start_datum']<'2019-04-01') & (loc_531['start_datum']>'2019-02-28')),:]

loc_531_march_2017=loc_531.loc[((loc_531['start_datum']<'2017-04-01') & (loc_531['start_datum']>'2017-02-28')),:]
loc_531_march_2018=loc_531.loc[((loc_531['start_datum']<'2018-04-01') & (loc_531['start_datum']>'2018-02-28')),:]
loc_531_march_2019=loc_531.loc[((loc_531['start_datum']<'2019-04-01') & (loc_531['start_datum']>'2019-02-28')),:]


loc_531_march_2017_pivot=loc_531_march_2017.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
loc_531_march_2018_pivot=loc_531_march_2018.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')
loc_531_march_2019_pivot=loc_531_march_2019.pivot_table(columns='start_tijd',values='waarnemingen_intensiteit',index='start_datum')

    
# 2017
for index, row in loc_531_march_2017_pivot.iterrows():
    fig,ax=plt.subplots()
    ax.plot(row,color=siemens_groen)
    ax.set_xticks(np.linspace(0,23,24))
    ax.set_xticklabels(my_xticks,rotation=80)#,fontsize=8)
    ax.set_xlabel('Time of the day [hh:m:ss]')
    ax.set_title('Traffic flow {}'.format(index))
    ax.set_ylim((0,2000))
    fig.show()
  
#2018
for index, row in loc_531_march_2018_pivot.iterrows():
    fig,ax=plt.subplots()
    ax.plot(row,color=siemens_groen)
    ax.set_xticks(np.linspace(0,23,24))
    ax.set_xticklabels(my_xticks,rotation=80)#,fontsize=8)
    ax.set_xlabel('Time of the day [hh:m:ss]')
    ax.set_title('Traffic flow {}'.format(index))
    ax.set_ylim((0,2000))
    fig.show()

#2019       
for index, row in loc_531_march_2019_pivot.iterrows():
    fig,ax=plt.subplots()
    ax.plot(row,color=siemens_groen)
    ax.set_xticks(np.linspace(0,23,24))
    ax.set_xticklabels(my_xticks,rotation=80)#,fontsize=8)
    ax.set_xlabel('Time of the day [hh:m:ss]')
    ax.set_title('Traffic flow {}'.format(index))
    ax.set_ylim((0,2000))
    fig.show()

#%%First day of march
trans_y_true_test_531_march1=np.transpose(trans_y_true_test_531[1272:1273])
i=1272
for index, row in loc_531_march_2019_pivot.iterrows():
    fig,ax=plt.subplots()
    ax.plot(row,color=o1)
    ax.plot(np.transpose(trans_y_true_test_531[i:i+1]),color=o1,label='True')
    ax.plot(np.transpose(trans_y_pred_test_531[i:i+1]),color=siemens_groen,label='Transformer')
    ax.set_xticks(np.linspace(0,23,24))
    ax.set_xticklabels(my_xticks,rotation=80)#,fontsize=8)
    ax.set_xlabel('Time of the day [hh:m:ss]')
    ax.set_title('Traffic flow {}'.format(index))
    ax.set_ylim((0,2000))
    ax.legend()
    fig.show()
    i+=24
    
    
#%% Make a prediction of a week with uncertainty ranges

# 1. Transformer 501, make easy prediction week of 2 sep 24 hours ahead.
start=151-2-3+2-1+7+7+7+7+7+7
start_2sep=31+28+31+30+31+30+31+31+1+(-2)+(-4)
start_26aug=31+28+31+30+31+30+31+25+(-2)+(-4)
start_17juni=31+28+31+30+31+16+(-2)+(-4)
start_24juni=31+28+31+30+31+23+(-2)+(-4)
start_3april=31+28+31+2+(-2)+(-2)
start_7jan=6-2-1

loc_ticks=np.array([12,12+24,12+2*24,12+3*24,12+4*24,12+5*24,12+6*24])

x_ticks_26aug=['08-26-2019','08-27-2019','08-28-2019','08-29-2019','08-30-2019','08-31-2019','09-01-2019']
x_ticks_2sep=['09-02-2019','09-03-2019','09-04-2019','09-05-2019','09-06-2019','09-07-2019','09-08-2019']
x_ticks_7jan=['01-07-2019','01-08-2019','01-09-2019','01-10-2019','01-11-2019','01-12-2019','01-13-2019']

ahead=0
start_index=start_2sep*24-ahead
days_to_predicht=7

#calculate percentile lines
percentile_5_trans_501_tile=np.tile(percentile_5_trans_501,(353,1))
trans_pred_501_5_line=trans_y_pred_test_501*percentile_5_trans_501_tile

percentile_25_trans_501_tile=np.tile(percentile_25_trans_501,(353,1))
trans_pred_501_25_line=trans_y_pred_test_501*percentile_25_trans_501_tile

percentile_75_trans_501_tile=np.tile(percentile_75_trans_501,(353,1))
trans_pred_501_75_line=trans_y_pred_test_501*percentile_75_trans_501_tile

percentile_95_trans_501_tile=np.tile(percentile_95_trans_501,(353,1))
trans_pred_501_95_line=trans_y_pred_test_501*percentile_95_trans_501_tile

x = np.linspace(0,24*days_to_predicht-1,24*days_to_predicht)

fig,ax =plt.subplots()
ax.fill_between(x, trans_pred_501_5_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_95_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, trans_pred_501_25_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_75_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=trans_y_pred_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='Transformer')
ax=sns.lineplot(x=x, y=trans_y_true_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_2sep,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501 at a horizon of 24h')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

# %%2. Transformer 501, make easy prediction week of 2 sep 1 hours ahead.
ahead=0
start_index=start_2sep*24-ahead
days_to_predicht=7

percentile_5_trans_501_tile=np.tile(percentile_5_trans_501,(353,1))
trans_pred_501_5_line=trans_y_pred_test_501*percentile_5_trans_501_tile

percentile_25_trans_501_tile=np.tile(percentile_25_trans_501,(353,1))
trans_pred_501_25_line=trans_y_pred_test_501*percentile_25_trans_501_tile

percentile_75_trans_501_tile=np.tile(percentile_75_trans_501,(353,1))
trans_pred_501_75_line=trans_y_pred_test_501*percentile_75_trans_501_tile

percentile_95_trans_501_tile=np.tile(percentile_95_trans_501,(353,1))
trans_pred_501_95_line=trans_y_pred_test_501*percentile_95_trans_501_tile

x = np.linspace(0,24*days_to_predicht-1,24*days_to_predicht)

fig,ax =plt.subplots()
ax.fill_between(x, trans_pred_501_5_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_95_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, trans_pred_501_25_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_75_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=trans_y_pred_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='Transformer')
ax=sns.lineplot(x=x, y=trans_y_true_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_2sep,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501 at a horizon of 1h')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

ahead=23
start_index=start_7jan*24-ahead
days_to_predicht=7

percentile_5_trans_501_tile=np.tile(percentile_5_trans_501,(353,1))
trans_pred_501_5_line=trans_y_pred_test_501*percentile_5_trans_501_tile

percentile_25_trans_501_tile=np.tile(percentile_25_trans_501,(353,1))
trans_pred_501_25_line=trans_y_pred_test_501*percentile_25_trans_501_tile

percentile_75_trans_501_tile=np.tile(percentile_75_trans_501,(353,1))
trans_pred_501_75_line=trans_y_pred_test_501*percentile_75_trans_501_tile

percentile_95_trans_501_tile=np.tile(percentile_95_trans_501,(353,1))
trans_pred_501_95_line=trans_y_pred_test_501*percentile_95_trans_501_tile

x = np.linspace(0,24*days_to_predicht-1,24*days_to_predicht)

fig,ax =plt.subplots()
ax.fill_between(x, trans_pred_501_5_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_95_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, trans_pred_501_25_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_75_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=trans_y_pred_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='Transformer')
ax=sns.lineplot(x=x, y=trans_y_true_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_7jan,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501 at a horizon of 24h')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))
plt.savefig(r"C:\UserData\z0049unj\Documents\Afstuderen\python\Simple models\figures_results\pred_trans_7jan_24h.pdf",bbox_inches='tight')

ahead=0
start_index=start_7jan*24-ahead
days_to_predicht=7

percentile_5_trans_501_tile=np.tile(percentile_5_trans_501,(353,1))
trans_pred_501_5_line=trans_y_pred_test_501*percentile_5_trans_501_tile

percentile_25_trans_501_tile=np.tile(percentile_25_trans_501,(353,1))
trans_pred_501_25_line=trans_y_pred_test_501*percentile_25_trans_501_tile

percentile_75_trans_501_tile=np.tile(percentile_75_trans_501,(353,1))
trans_pred_501_75_line=trans_y_pred_test_501*percentile_75_trans_501_tile

percentile_95_trans_501_tile=np.tile(percentile_95_trans_501,(353,1))
trans_pred_501_95_line=trans_y_pred_test_501*percentile_95_trans_501_tile

x = np.linspace(0,24*days_to_predicht-1,24*days_to_predicht)

fig,ax =plt.subplots()
ax.fill_between(x, trans_pred_501_5_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_95_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, trans_pred_501_25_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_75_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=trans_y_pred_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='Transformer')
ax=sns.lineplot(x=x, y=trans_y_true_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_7jan,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501 at a horizon of 1h')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))


ahead=23
start_index=start_26aug*24-ahead
days_to_predicht=7

percentile_5_trans_501_tile=np.tile(percentile_5_trans_501,(353,1))
trans_pred_501_5_line=trans_y_pred_test_501*percentile_5_trans_501_tile

percentile_25_trans_501_tile=np.tile(percentile_25_trans_501,(353,1))
trans_pred_501_25_line=trans_y_pred_test_501*percentile_25_trans_501_tile

percentile_75_trans_501_tile=np.tile(percentile_75_trans_501,(353,1))
trans_pred_501_75_line=trans_y_pred_test_501*percentile_75_trans_501_tile

percentile_95_trans_501_tile=np.tile(percentile_95_trans_501,(353,1))
trans_pred_501_95_line=trans_y_pred_test_501*percentile_95_trans_501_tile

x = np.linspace(0,24*days_to_predicht-1,24*days_to_predicht)

fig,ax =plt.subplots()
ax.fill_between(x, trans_pred_501_5_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_95_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, trans_pred_501_25_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_75_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=trans_y_pred_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='Transformer')
ax=sns.lineplot(x=x, y=trans_y_true_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501 at a horizon of 24h')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

ahead=0
start_index=start_26aug*24-ahead
days_to_predicht=7

percentile_5_trans_501_tile=np.tile(percentile_5_trans_501,(353,1))
trans_pred_501_5_line=trans_y_pred_test_501*percentile_5_trans_501_tile

percentile_25_trans_501_tile=np.tile(percentile_25_trans_501,(353,1))
trans_pred_501_25_line=trans_y_pred_test_501*percentile_25_trans_501_tile

percentile_75_trans_501_tile=np.tile(percentile_75_trans_501,(353,1))
trans_pred_501_75_line=trans_y_pred_test_501*percentile_75_trans_501_tile

percentile_95_trans_501_tile=np.tile(percentile_95_trans_501,(353,1))
trans_pred_501_95_line=trans_y_pred_test_501*percentile_95_trans_501_tile

x = np.linspace(0,24*days_to_predicht-1,24*days_to_predicht)

fig,ax =plt.subplots()
ax.fill_between(x, trans_pred_501_5_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_95_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, trans_pred_501_25_line[int(start_index):int(start_index+days_to_predicht*24),ahead], trans_pred_501_75_line[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=trans_y_pred_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='Transformer')
ax=sns.lineplot(x=x, y=trans_y_true_test_501[int(start_index):int(start_index+days_to_predicht*24),ahead],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501 at a horizon of 1h')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

#%%MAke prediction with uncertainty ranges for the baseline models
percentile_5_rf_501_tile=np.tile(percentile_5_rf_501,(356,1))
rf_pred_501_5_line=rf_y_pred_test_501*percentile_5_rf_501_tile

percentile_25_rf_501_tile=np.tile(percentile_25_rf_501,(356,1))
rf_pred_501_25_line=rf_y_pred_test_501*percentile_25_rf_501_tile

percentile_75_rf_501_tile=np.tile(percentile_75_rf_501,(356,1))
rf_pred_501_75_line=rf_y_pred_test_501*percentile_75_rf_501_tile

percentile_95_rf_501_tile=np.tile(percentile_95_rf_501,(356,1))
rf_pred_501_95_line=rf_y_pred_test_501*percentile_95_rf_501_tile

percentile_5_mlp_501_tile=np.tile(percentile_5_mlp_501,(356,1))
mlp_pred_501_5_line=mlp_y_pred_test_501*percentile_5_mlp_501_tile

percentile_25_mlp_501_tile=np.tile(percentile_25_mlp_501,(356,1))
mlp_pred_501_25_line=mlp_y_pred_test_501*percentile_25_mlp_501_tile

percentile_75_mlp_501_tile=np.tile(percentile_75_mlp_501,(356,1))
mlp_pred_501_75_line=mlp_y_pred_test_501*percentile_75_mlp_501_tile

percentile_95_mlp_501_tile=np.tile(percentile_95_mlp_501,(356,1))
mlp_pred_501_95_line=mlp_y_pred_test_501*percentile_95_mlp_501_tile

start_26aug_bl=(start_26aug+2)*24

#Random forest
fig,ax =plt.subplots()
ax.fill_between(x, rf_pred_501_5_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], rf_pred_501_95_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, rf_pred_501_25_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], rf_pred_501_75_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=rf_y_pred_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='Random forest')
ax=sns.lineplot(x=x, y=rf_y_true_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

#MLP
fig,ax =plt.subplots()
ax.fill_between(x, mlp_pred_501_5_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], mlp_pred_501_95_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, mlp_pred_501_25_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], rf_pred_501_75_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=mlp_y_pred_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='MLP')
ax=sns.lineplot(x=x, y=mlp_y_true_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

start_2sep_bl=(start_24juni+2)*24

#Random forest
fig,ax =plt.subplots()
ax.fill_between(x, rf_pred_501_5_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0], rf_pred_501_95_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_groen_light2,label='5%-95%')
ax.fill_between(x, rf_pred_501_25_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0], rf_pred_501_75_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=rf_y_pred_test_501[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_blauw,label='Random forest')
ax=sns.lineplot(x=x, y=rf_y_true_test_501[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

#MLP
fig,ax =plt.subplots()
ax.fill_between(x, mlp_pred_501_5_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0], mlp_pred_501_95_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, mlp_pred_501_25_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0], rf_pred_501_75_line[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=mlp_y_pred_test_501[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_blauw,label='MLP')
ax=sns.lineplot(x=x, y=mlp_y_true_test_501[int(start_2sep_bl):int(start_2sep_bl+7*24),0],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

start_26aug_bl=(start_7jan+2)*24

#Random forest
fig,ax =plt.subplots()
ax.fill_between(x, rf_pred_501_5_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], rf_pred_501_95_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, rf_pred_501_25_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], rf_pred_501_75_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=rf_y_pred_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='Random forest')
ax=sns.lineplot(x=x, y=rf_y_true_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))

#MLP
fig,ax =plt.subplots()
ax.fill_between(x, mlp_pred_501_5_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], mlp_pred_501_95_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen_light2,label='5%-95%'); 
ax.fill_between(x, mlp_pred_501_25_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0], rf_pred_501_75_line[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_groen,label='25%-75%')
ax=sns.lineplot(x=x, y=mlp_y_pred_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='MLP')
ax=sns.lineplot(x=x, y=mlp_y_true_test_501[int(start_26aug_bl):int(start_26aug_bl+7*24),0],color=siemens_blauw,label='True',linestyle='--')
ax.set_xticks(loc_ticks)
ax.set_xticklabels(x_ticks_26aug,rotation=30)
plt.legend(loc=1)
plt.title('Traffic flow prediction for location 501')
plt.xlabel('Date')
plt.ylabel('Traffic flow [veh/h]')
plt.ylim((0,1900))


