# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:36:12 2022

@author: z0049unj
"""

#%% Import packages
import pandas as pd
import numpy as np
from scipy import stats # for z-score normalization
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope

#Define colors 
siemens_groen=(0/255,153/255,153/255)
o7=(255/255, 193/255, 158/255)
o1=(242/255, 98/255, 0/255)
siemens_groen_light1=(92/255,176/255,175/255)
siemens_groen_light2=(142/255,198/255,197/255)
siemens_groen_light3=(188/255,221/255,220/255)
siemens_blauw=(1/255,8/255,46/255)
siemens_blauw_groen=(0/255,90/255,120/255)
wit=(1,1,1)


# %% Read data
look_back=2*24
pred_horizon=24
d_model=11

#Original index before data preprocessing
original_index_501=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Original_index_before_preprocess/original_index_501.pkl")


def import_data_location(location):
    loc=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Simple models/Location_observation_data/loc_{}.pkl".format(location))
    X_train=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/X_{}_train_final.pkl".format(location))
    X_test=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/X_{}_test_final.pkl".format(location))
    y_train=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/y_{}_train.pkl".format(location))
    y_test=pd.read_pickle("C:/UserData/z0049unj/Documents/Afstuderen/python/Final_Feature_Set/y_{}_test.pkl".format(location))
    #info about y to scale back output and train on standardized set
    y_train_stand_values=stats.zscore(y_train.values).reshape(y_train.shape[0],)
    y_train_norm=(y_train-y_train.min())/(y_train.max()-y_train.min()) #min max normalization
    y_train_stand=pd.DataFrame({'waarnemingen_intensiteit':y_train_stand_values})

    
    
    y_train_mean=np.mean(y_train.values)
    y_train_std=np.std(y_train.values)
    
    y_test_stand=(y_test-y_train_mean)/y_train_std

    
    return loc,X_train,X_test,y_train,y_test,y_train_stand,y_test_stand,y_train_norm,y_train_mean,y_train_std

[loc_501,X_501_train,X_501_test,y_501_train,y_501_test,y_501_train_stand,y_501_test_stand,y_501_train_norm,y_501_mean,y_501_std]=import_data_location('501')


# %%Set up final feature set
X_501_train_time=X_501_train.copy(deep=True) 
X_501_test_time=X_501_test.copy(deep=True)

del X_501_train_time['Sun_duration']
del X_501_test_time['Sun_duration']

# Add additional season feature
days_in_year=np.linspace(0,364,365)
sin_time_season=np.sin(2*np.pi*days_in_year/365) 
n_years=3
sin_time_season_three_years=[]
for i in range(n_years):
    sin_time_season_three_years=np.append(sin_time_season_three_years,sin_time_season)

all_days_array=np.repeat(sin_time_season_three_years,24) #because all 24 hours in the day have the same value
all_day_df=pd.DataFrame(data={'all_days':all_days_array})

season_nodig=all_day_df.loc[original_index_501['original_index'],:].reset_index(drop=True)


# Concatenate train and test data
X_total=pd.concat([X_501_train_time,X_501_test_time])
X_total['season2']=season_nodig
y_total=pd.concat([y_501_train_stand,y_501_test_stand])

# Add school holiday as a feature
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

# Split in the test and train set
X_train_option2=X_total.loc[X_total.index[loc_501['start_datum']<'2019-01-01'],:]
X_test_option2=X_total.loc[X_total.index[loc_501['start_datum']>'2018-12-31'],:]

y_train_option2=y_total.loc[y_total.index[loc_501['start_datum']<'2019-01-01'],:]
y_test_option2=y_total.loc[y_total.index[loc_501['start_datum']>'2018-12-31'],:]

# Final Feature set
X_train_option2=pd.DataFrame({'sin_time':X_train_option2['sin_time'],'cos_time':X_train_option2['cos_time'],'mon':X_train_option2['mon'], 'tue':X_train_option2['tue'], 'wed':X_train_option2['wed'], 'thu':X_train_option2['thu'], 'fri':X_train_option2['fri'], 'sat':X_train_option2['sat'], 'sun':X_train_option2['sun'],'season1':X_train_option2['season'],'season2':X_train_option2['season2'],'feestdag':X_train_option2['feestdag'],'vacation':X_train_option2['vacation'],'Temperature':X_train_option2['Temperature'],'Rel_humidity':X_train_option2['Rel_humidity'],'Radiation':X_train_option2['Radiation']})
X_test_option2=pd.DataFrame({'sin_time':X_test_option2['sin_time'],'cos_time':X_test_option2['cos_time'],'mon':X_test_option2['mon'], 'tue':X_test_option2['tue'], 'wed':X_test_option2['wed'], 'thu':X_test_option2['thu'], 'fri':X_test_option2['fri'], 'sat':X_test_option2['sat'], 'sun':X_test_option2['sun'],'season1':X_test_option2['season'],'season2':X_test_option2['season2'],'feestdag':X_test_option2['feestdag'],'vacation':X_test_option2['vacation'],'temperature':X_test_option2['Temperature'],'Rel_humidity':X_test_option2['Rel_humidity'],'Radiation':X_test_option2['Radiation']})

#%% Create data encoder and decoder input data sets
def create_dataset_encdec(dataset,xdata,look_back,pred_horizon):
    Encoder_input, Decoder_input, Decoder_output= [],[],[]
    y=dataset.values
    x=xdata.values
    for i in range(len(dataset)-look_back-pred_horizon):
        flow_input      =   y[i:(i+look_back),:]
        time_input      =   x[i:(i+look_back),:]
        encoder_input_data = np.column_stack((flow_input,time_input))
        Encoder_input.append(encoder_input_data)
        
        flow_output=y[(i+look_back):(i+look_back+pred_horizon),0]
        Decoder_output.append(flow_output)
        
        flow_input_decoder=y[(i+look_back-1):(i+look_back+pred_horizon-1),:]
        time_input_decoder=x[(i+look_back):(i+look_back+pred_horizon),:]
        decoder_input_data=np.column_stack((flow_input_decoder,time_input_decoder))
        Decoder_input.append(decoder_input_data)       
                
    return np.array(Encoder_input,dtype=np.float32),np.array(Decoder_input,dtype=np.float32),np.array(Decoder_output,dtype=np.float32)

#Train set
Enc_input_501, Dec_input_501, Dec_output_501=create_dataset_encdec(y_train_option2,X_train_option2,look_back=look_back,pred_horizon=pred_horizon)
Dec_output_501=Dec_output_501.reshape(Dec_output_501.shape[0],Dec_output_501.shape[1],1)
# Test set
Enc_input_501_test, Dec_input_501_test, Dec_output_501_test=create_dataset_encdec(y_test_option2,X_test_option2,look_back=look_back,pred_horizon=pred_horizon)


# %% Positional encoding
def positional_encoding(x, horizon):
    
    # One dimensional positional encoding
    pos_encoding_1d=np.linspace(0,1,horizon)
    pos_encoding_1d=pos_encoding_1d.reshape(pos_encoding_1d.shape[0],1)
    pos_toadd=tf.cast(pos_encoding_1d, dtype=tf.float32)
    
    # Reshape to use in tile
    pos_to_add_multiple=tf.reshape(pos_toadd,(1,horizon,1))
    
    # Ccalculate number of inputs
    dimensie_input=tf.constant([x.shape[0],1,1], tf.int32)        
    
    #Concatenate positional encoding for each input sample
    pos_multiple=tf.tile(pos_to_add_multiple,dimensie_input)
    x=tf.concat((x,pos_multiple),axis=2) #dimensie positional encoding is gebaseerd op d_model
    
    return x
#Train set
Enc_input_501_pos=positional_encoding(Enc_input_501,look_back)
Dec_input_501_pos=positional_encoding(Dec_input_501,pred_horizon)
#Test set
Enc_input_501_pos_test=positional_encoding(Enc_input_501_test,look_back)
Dec_input_501_pos_test=positional_encoding(Dec_input_501_test,pred_horizon)

# %%Shuffle data
# Misschien nice om eerst te shuffelen anders niet echt eerlijk

np.random.seed(42)
shuffled_indices=np.random.permutation(len(Enc_input_501_pos))

Enc_input_501_pos_shuffled=tf.gather(Enc_input_501_pos,shuffled_indices)
Dec_input_501_pos_shuffled=tf.gather(Dec_input_501_pos,shuffled_indices)
Dec_output_501_shuffled=tf.gather(Dec_output_501,shuffled_indices)



#%% Define all components of the transformer

# 1. Look ahead mask function to be used in the decoder
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask

# 2. Scaled dot-product attention
def scaled_dot_product_attention(q, k, v, mask):

  dot_qk = tf.linalg.matmul(q, k, transpose_b=True)     # calculates the dot product and transposes the second argument by settint transpose_b=True
  dk = tf.cast(tf.shape(k)[-1], tf.float32)             # changes the variable into a float, and is equal to the dimension of the input feature (3 if flow and time)
  scaled_attention_logits = dot_qk / tf.math.sqrt(dk)   # scale the dot product
  
  # add the mask to the scaled tensor
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)            #approximately -infinity

  # softmax is used to normalized on the last axis (seq_len_k) so that the scores add up to 1
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) 
  output = tf.matmul(attention_weights, v)  

  return output, attention_weights

  
# 3. Multi head attention
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0        #remainder (modulus (%) equals zero)

    # if i do not want to decrease the input dimension just set depth equal to dimension (3 if 3 values)
    self.depth = d_model // self.num_heads  
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)
    

  def split_heads(self, x, batch_size):    
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))         #size -1 is computed such that the total size remains the same
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0] 
    # Define q
    q = self.wq(q)                                                          # (batch_size, seq_len, d_model)    
    k = self.wk(k)                                                          # (batch_size, seq_len, d_model)
    v = self.wv(v)                                                          # (batch_size, seq_len, d_model)
    
    # Make multiple heads and q,k,v of appropriate dimension
    q = self.split_heads(q, batch_size)                                     # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)                                     # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)                                     # (batch_size, num_heads, seq_len_v, depth)
    
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
    
    #Change into batchsize, seq length, num heads, dimension/heads   
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])    # (batch_size, seq_len_q, num_heads, depth)

    #Change back to batchsize,seq length, dimension
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)                                   # (batch_size, seq_len_q, d_model) concatenate different heads

    return output, attention_weights


# 4. Define Feedforward neural network
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),                        # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)                                        # (batch_size, seq_len, d_model)
  ])

#5. ENCODER LAYER
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)                       # multihead attention
    self.ffn = point_wise_feed_forward_network(d_model, dff)                # feedforwards nn

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)      # first layer normalization
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)      # second layer normalization

    self.dropout1 = tf.keras.layers.Dropout(rate)                           # first dropout layer
    self.dropout2 = tf.keras.layers.Dropout(rate)                           # second dropout layer

  def call(self, x, training, mask):
    # set up paper: MHA --> dropout --> normalization --> FNN --> dropout --> normalization  
    attn_output, attention_weight = self.mha(x, x, x, mask)                 # (batch_size, input_seq_len, d_model) the input x is set to the v,k,q 
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)                                 # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)                                             # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)                               # (batch_size, input_seq_len, d_model)
    return out2,attention_weight



# 6. DECODER LAYER
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,look_ahead_mask, padding_mask):
    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)        # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    
    out1 = self.layernorm1(attn1 + x)
    # Set padding_mask to none, because not required,if  sequences of constant length
    attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # output decoder 1 is the query, encoder output key and value
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)                                    # (batch_size, target_seq_len, d_model)
  
    ffn_output = self.ffn(out2)                                             # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)                               # (batch_size, target_seq_len, d_model)
    return out3, attn_weights_block1, attn_weights_block2
    

# 7. Encoder
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff,  rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    attention_weights = {}
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
      x, block1 = self.enc_layers[i](x, training, mask)
      attention_weights[f'encoder_layer{i+1}_block1'] = block1
      
    return x,attention_weights                                              # (batch_size, input_seq_len, d_model)

# 8. Decoder
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model,  num_heads, dff, target_vocab_size,  rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,look_ahead_mask, padding_mask):
    
    attention_weights = {}

    
    x = self.dropout(x, training=training)
    for i in range(self.num_layers):
    # output of decoder layer final output, attention weights 1 and attention weights 2
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights



# 9. Transformer
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers, d_model,  num_heads, dff, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
  def call(self, inputs, training):
    inp, tar = inputs
    look_ahead_mask = self.create_masks(inp, tar)
    enc_output,attention_weight_enc = self.encoder(inp, training,mask=None)         # (batch_size, inp_seq_len, d_model)
    dec_output, attention_weights = self.decoder( tar, enc_output, training, look_ahead_mask,padding_mask=None)
    
    final_output = self.final_layer(dec_output)                                     # (batch_size, tar_seq_len, target_vocab_size)
    return final_output,attention_weights,attention_weight_enc,enc_output           # investigate wheter weights have to be returned
        


  def create_masks(self, inp, tar):
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])

    return look_ahead_mask

# %%Bayesian hyperparameter optimization

# 1. Define non hyperparameters
d_model=Enc_input_501_pos.shape[2]
target_vocab_size=1 
dropout_rate=0

# 2. Make hyperparameter space
head_options_initial=[1,2,4] # Only options for time1 time 2, flow and age, because just 4 dimensions
head_options_total=[1,2,3,6,9,18]

space={ 
       'num_heads':hp.choice('num_heads',head_options_initial),
       'num_layers': scope.int(hp.quniform('n_layers_enc',1,10,1)), #quniform returns float however int required  therefore scope.int (returns for example 4.0)
       'dff': scope.int(hp.quniform('dimension_enc',10,400,10)),
       'learning_rate': hp.choice('learning_rate',[0.00001,0.0001,0.001, 0.01]),
       'batch_size':hp.choice('batch_size',[16,32,64]),
       }

# 3. Build model
def build_transformer_opt(params):
    Enc_input=Enc_input_501_pos_shuffled
    Dec_input=Dec_input_501_pos_shuffled
    Dec_output=Dec_output_501_shuffled
    
    d_model_def=d_model
    target_vocab_size_def=target_vocab_size
    dropout_rate_def=dropout_rate
    
    transformer_model=Transformer(
        num_layers=params['num_layers'],
        d_model=d_model_def,
        num_heads=params['num_heads'],  
        dff=params['dff'],
        target_vocab_size=target_vocab_size_def,
        rate=dropout_rate_def)
    
    Optimizer=keras.optimizers.Adam(lr=params['learning_rate'])
    loss_part1=['mse']
    loss_part2=[None]*(2*params['num_layers'])
    loss_part1.extend(loss_part2)
    print(loss_part1)
    transformer_model.compile(optimizer=Optimizer, loss=loss_part1)#,None,None,None,None,None,None])
    
    es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=15)    

    result=transformer_model.fit([Enc_input,Dec_input],Dec_output,epochs=40,validation_split=0.2,batch_size=params['batch_size'],callbacks=[es])

    validation_loss = np.amin(result.history['val_loss']) 
    print('Best validation loss of epoch:', validation_loss)
    
    return {'loss': validation_loss,   
            'status': STATUS_OK, 
            'model': transformer_model, 
            'params': params}

# 4. Pass the model to the optimization
trials =Trials() #saves everything
best=fmin(build_transformer_opt,
          space,
          algo=tpe.suggest,
          max_evals=100, # maybe 50 or so, just small to try now
          trials=trials
          )

# 5. evaluate
best_model = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['model']
best_params = trials.results[np.argmin([r['loss'] for r in 
    trials.results])]['params']
worst_model = trials.results[np.argmax([r['loss'] for r in 
    trials.results])]['model']
worst_params = trials.results[np.argmax([r['loss'] for r in 
    trials.results])]['params']

# %% Transformer implementation 
#hyperparameters 
num_layers =4           
d_model = Enc_input_501_pos.shape[2]
dff =360              
num_heads = 3       
dropout_rate = 0.2
target_vocab_size=1     
look_back=look_back
horizon=pred_horizon
learning_rate=0.0001   
batch_size=16

# Define
transformer_optie2= Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,  
    dff=dff,
    target_vocab_size=target_vocab_size,
    rate=dropout_rate)


Optimizer=keras.optimizers.Adam(lr=learning_rate)

# Train
#Required to set None to ensure that only trained on output flow and not on attentionweights, 2*number of layers
transformer_optie2.compile(optimizer=Optimizer, loss=["mse",None,None,None,None,None,None,None,None])
result=transformer_optie2.fit([Enc_input_501_pos_shuffled,Dec_input_501_pos_shuffled],Dec_output_501_shuffled,epochs=120,batch_size=batch_size,validation_split=0.2)

transformer_optie2.summary()

# Plot the learning curve during training result.hisotry return losses during epochs
plt.figure()
plt.plot(result.history['loss'],label='training loss',color=siemens_groen)
plt.plot(result.history['val_loss'],label='validation loss',color=o1)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('MSE [$veh^2/h^2$]')
plt.title("Learning curves for location 531 ")
plt.ylim((0,0.3))

#%% Predict
class Predictor(tf.Module):
  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, first_flow ,known_input,start_seq,n_pred,pred_horizon,Encoder_input): # first flow=y(t), known_input=x(t+1)--x(t+24)
    
    print('known input',known_input.shape)
    # from here loop, so start with the first, different steps due to other dimensions
    known_input_first=known_input[:,0:1,:]            # extract the frist known inputs, 0:1 such that stays 3D   
    print('known_input_first',known_input_first.shape)
    # first decoder input to predict y(t+1), based on y(t) and x(t+1)
    Dec_input=tf.concat((first_flow,known_input_first),axis=2) # concatenate first flow and inputs to dimensie 1,1,4 
    print('Dec_input',Dec_input.shape)
    for i in range(pred_horizon-1):
        print(i)
        # make prediction y(t+1)
        predicted,_,_,_=self.transformer ([Encoder_input[start_seq:(start_seq+n_pred),:,:], Dec_input], training=False) # dimensie 1,1,1
        
        # concatenate first flow with predicted
        known_flow=tf.concat((first_flow,predicted),axis=1) # concatenate first flow and inputs to dimensie 1,1,4
        # extract known input information x(t=1) and x(t+2)
        known_input_part=known_input[:,0:(i+2),:]   
        
        # Concatenate known input and flows
        Dec_input= tf.concat((known_flow,known_input_part),axis=2) 
    
    # final prediction
    predicted_final,attention_weights,attention_weight_enc,enc_output=self.transformer ([Encoder_input[start_seq:(start_seq+n_pred),:,:], Dec_input], training=False) # dimensie 1,1,1
    # I can extract last attention weights here aswell, is nice to see


    return predicted_final,attention_weights,attention_weight_enc,enc_output


#%% Make prediction for train, val, and test set
# 1. Predict on entire train and validation set
# define predictor from transformer test 
predictor_train=Predictor(transformer_optie2)
predictor_test=Predictor(transformer_optie2)

# Set up inputs for the prediction
start_seq=0                                             # which index to start prediction sequence
n_pred_train=Enc_input_501_pos_shuffled.shape[0]        # numer of predictions
n_pred_test=Enc_input_501_pos_test.shape[0]             # numer of predictions


first_flow=Enc_input_501_pos_shuffled[start_seq:(start_seq+n_pred_train),-1,0]          # we need the first flow for the first prediction y(t+1)
first_flow=tf.reshape(first_flow,(n_pred_train,1,1))                                    # reshape to three dimensional tensor
known_input=Dec_input_501_pos_shuffled[start_seq:(start_seq+n_pred_train),:,1:]         # the entire known input, so without the flow

# Make the prediction train and validation set
y_pred,weights,weights_enc,enc_output=predictor_train(first_flow,known_input,start_seq,n_pred_train,pred_horizon,Enc_input_501_pos_shuffled)
dimension_train=int(y_pred.shape[0]*0.8)
y_pred_train=y_pred[0:dimension_train,:]
y_pred_val=y_pred[dimension_train:,:]

first_flow=Enc_input_501_pos_test[start_seq:(start_seq+n_pred_test),-1,0]      # we need the first flow for the first prediction y(t+1)
first_flow=tf.reshape(first_flow,(n_pred_test,1,1))       # reshape to three dimensional tensor
known_input=Dec_input_501_pos_test[start_seq:(start_seq+n_pred_test),:,1:]     # the entire known input, so without the flow

# Make the prediction for the test set
y_pred_test,weights_test,weights_enc_test,enc_output_test=predictor_test(first_flow,known_input,start_seq,n_pred_test,pred_horizon,Enc_input_501_pos_test)


