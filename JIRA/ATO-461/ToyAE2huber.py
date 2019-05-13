
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, GaussianNoise, Concatenate, Lambda, Subtract
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from functools import partial


# In[2]:


MCdata = pd.read_csv("MC_data3.csv")
MCdata =MCdata.sample(50000)

# In[5]:


def custom_loss(y_true, y_pred):

    d = (y_true - y_pred)/ y_true 
    d0=d[:,0]
    d1=d[:,1]*10
    d2=d[:,2]
    d3=d[:,3]
    d4=d[:,4]
    d5=d[:,5]
    
    def huber(x):
        db = 3
        xsq = x*x
        term1= 0.5* xsq 
        term2= 0.5* db*db + db * (x-db)
        return K.switch(x>db, term2, term1)

    def huberps(x):
        db = 3
        xsq = x*x
        x = K.abs(x)
        term1 = xsq
        term2 = db*db + x - db
        return K.switch(x>db, term2, term1)
    
    
    return (huber(d0)+huber(d1)+huber(d2)+huber(d3)+huber(d4)+huber(d5))/6
# to add regulaizers - model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))

inputs = Input(shape=(7,))
#inputw = Input(shape=(6,))
inputlnp = Input(shape=(1,))
#noise = GaussianNoise(0.2)(inputs)
enc1   = Dense(units=64, activation='selu')(inputs)
enc2   = Dense(units=32, activation='selu')(enc1)
enc3   = Dense(units=16, activation='selu')(enc2)
enc4   = Dense(units=8, activation='selu')(enc3)
enc5   = Dense(units=8, activation='selu')(enc4)
#enc9   = Dense(units=64, activation='selu')(enc8)
#enc10   = Dense(units=64, activation='selu')(enc9)
#enc11   = Dense(units=64, activation='selu')(enc10)
#enc12   = Dense(units=64, activation='selu')(enc11)
#enc13   = Dense(units=64, activation='selu')(enc12)
lnm = Dense(units=1 , activation='linear')(enc5)
gb     = Subtract()([inputlnp,lnm])
#expgb = Lambda(lambda x: K.exp(x))(gb)
dec1   = Dense(units=8, activation='selu')(gb)
dec2   = Dense(units=8, activation='selu')(dec1)
dec3   = Dense(units=16, activation='selu')(dec2)
dec4   = Dense(units=32, activation='selu')(dec3)
dec5   = Dense(units=64, activation='selu')(dec4)
#dec9   = Dense(units=64, activation='selu')(dec8)
#dec10   = Dense(units=64, activation='selu')(dec9)
#dec11   = Dense(units=64, activation='selu')(dec10)
#dec12   = Dense(units=64, activation='selu')(dec11)
#dec13   = Dense(units=64, activation='selu')(dec12)
outputs= Dense(units=6, activation='linear')(dec5)

modelpt = Model(inputs=[inputs,inputlnp],outputs=outputs)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.00001)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
modelpt.compile(loss=custom_loss,
              optimizer=opt,
              metrics=['mse'])

modelpt.summary()



# In[7]:


train, test =train_test_split(MCdata, test_size=0.5)

train_pt = train[["ITS", "TOF", "TPCROC0", "TPCROC1", "TPCROC2", "TRD", "pmeas"]]
test_pt = test[["ITS", "TOF", "TPCROC0", "TPCROC1", "TPCROC2", "TRD", "pmeas"]]
scaler_pt = StandardScaler()
scaler_pt.fit( train[["ITS", "TOF", "TPCROC0", "TPCROC1", "TPCROC2", "TRD", "pmeas"]])

scaler_pt.transform(train_pt)[:,:6]


# In[ ]:


modelpt.fit([scaler_pt.transform(train_pt),train["lnp"]], train[["ITS", "TOF", "TPCROC0", "TPCROC1", "TPCROC2", "TRD"]], epochs=50, batch_size=128, validation_data=[[scaler_pt.transform(test_pt),test["lnp"]],test[["ITS", "TOF", "TPCROC0", "TPCROC1", "TPCROC2", "TRD"]]])


# In[ ]:


modelpt.save('MCcheckhuber.h5')
#
#
#out = modelpt.predict([spidvalues,spvalue])
#out = pd.DataFrame(out)
#perfectdatacmplt = perfectdatacmplt.reset_index()
#outges = pd.concat([perfectdatacmplt, out],axis=1)
#outges.to_csv("output.csv")
