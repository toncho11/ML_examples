# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:06:08 2024

A transformer class for TimeVAE  

@author: antona
"""

import os
import glob
import time
import sys
import gc

sys.path.insert(1, 'C:/Work/PythonCode/ML_examples/AutoEncoders/TimeVAE') #should be changed to your TimeVAE code
#from local files:
import utils
from vae_dense_model import VariationalAutoencoderDense as VAE_Dense
from vae_conv_model import VariationalAutoencoderConv as VAE_Conv
from vae_conv_I_model import VariationalAutoencoderConvInterpretable as TimeVAE

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

#TimeVAE
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class TimeVaeTransformer(BaseEstimator, ClassifierMixin, TransformerMixin):

    def __init__(self,
                 ):
        """Init."""
        pass
        
    def fit(self, X, y):
        
        print("In TrainVAE fit")
        
        selected_class = 1 #TODO check if used and how
        iterations = 500
        hidden_layer_low = 500 
        latent_dim = 8
        onlyP300 = False
        
        if onlyP300 == True:
        
            selected_class_indices = np.where(y == selected_class)
            
            X = X[selected_class_indices,:,:]
        
            X = X[-1,:,:,:] #remove the first exta dimension
        
        print("Samples count used by the VAE train: ", X.shape)
        
        #FIX: not the correct format (3360, 8, 257) but should be (3360, 257, 8)
        N, T, D = X.shape #N = number of samples, T = time steps, D = feature dimensions
        print(N, T, D)
        # X = X.reshape(N,D,T)
        X = X.transpose(0,2,1)
        N, T, D = X.shape
        print(N, T, D)
        
        # min max scale the data    
        scaler = utils.MinMaxScaler()        
       
        scaled_data = scaler.fit_transform(X)
        
        #vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[200,100], )
        vae = VAE_Dense( seq_len=T,  feat_dim = D, latent_dim = latent_dim, hidden_layer_sizes=[hidden_layer_low * 2, hidden_layer_low], )
        
        vae.compile(optimizer=Adam())
        # vae.summary()

        early_stop_loss = 'loss'
        #define two callbacks
        #early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
        #reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced
        
        early_stop_callback = EarlyStopping(monitor=early_stop_loss, min_delta = 1e-1, patience=10) 
        reduceLR = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10)  #From TensorFLow: if no improvement is seen for a 'patience' number of epochs, the learning rate is reduced

        print("Fit VAE")
        vae.fit(
            scaled_data, 
            batch_size = 32, #default 32
            epochs=iterations, #default 500
            shuffle = True,
            #callbacks=[early_stop_callback, reduceLR],
            verbose = 0
        )
        
        self.vae = vae
        self.scaler = scaler

        return self
    
    def _encode(self,X):
        
        #timeVAE, scaler, X_train, X_test, y_train, y_test
        
        #There are 3 outputs of the VAE Encoder
        #encoder = [z_mean, z_log_var, encoder_output]
        #z_mean, z_log_var, z = self.encoder(data)
        
        #There are 3 outputs. We can use one of them as feature vector or all of them together.
        self.output = 2
        
        #Use the auto encoder to produce feature vectors
        
        # Process X_train
        X_trained_c = X.copy()
        
        # convert to correct format expected by timeVAE
        N, T, D = X_trained_c.shape #N = number of samples, T = time steps, D = feature dimensions
        print(N, T, D)
        X_trained_c = X_trained_c.transpose(0,2,1)
        N, T, D = X_trained_c.shape
        print(N, T, D)
        
        X_train_scaled = self.scaler.fit_transform(X_trained_c)
        
        X_train_fv = self.timeVAE.encoder.predict(X_train_scaled)
        
        #X_train_fv_np = np.array(X_train_fv)[ output ]    
        #version that use all 3 outputs from the encoder
        X_train_fv_all = []
        # for i in range(0, len(y_train)):
        #     X_train_fv_all.append(np.concatenate((X_train_fv[0][i], X_train_fv[1][i], X_train_fv[2][i])))
        
        for i in range(0, X.shape[0]): # len(y_train)
            X_train_fv_all.append(X_train_fv[self.output][i])
        
        X_train_fv_np = np.array(X_train_fv_all)  
        
        return X_train_fv_np    #returns encoded data using the trained TimeVAE 

    def transform(self, X,):
        
        X_tansformed = self._encode(X)
        
        return X_tansformed

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.tansform(X)