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
                 encoder_output = 3,
                 use_augmented_data = False,
                 vae_iterations = 5,
                 use_pretrained_scaler = True,
                 vae_latent_dim = 8
                 #onlyP300 = False
                 ):
        """Init."""
        
        #There are 3 outputs of the VAE Encoder
        #encoder = [z_mean, z_log_var, encoder_output]
        #z_mean, z_log_var, z = self.encoder(data)
        #There are 3 outputs. We can use one of them as feature vector or all of them together.
        ##hich output from the encoder to give to the transform method()
        #3 means all
        self.encoder_output = encoder_output
        self.use_augmented_data = use_augmented_data
        self.vae_iterations = vae_iterations
        self.use_pretrained_scaler = use_pretrained_scaler
        self.vae_latent_dim = vae_latent_dim
        #self.onlyP300 = onlyP300
    
    def _generateInterpolated(self, X, y , selected_class, shift = 0, every_ith_sample = 10): #shift the starting point in the epoch
        
        #print ("Generating interpolated samples...")
        
        Xint = X.copy()
        epoch_length = X.shape[2]
        channels     = X.shape[1]
        indicesSelectedClass = []
        
        for i in range(0, len(y)):
            if y[i] == selected_class:
                for j in range(shift, epoch_length, every_ith_sample):
                    if j - 1 > 0 and j + 1 < epoch_length:
                        for m in range(0,channels):
                            #modify it
                            Xint[i,m,j] = (X[i,m,j-1] + X[i,m,j+1]) / 2
                indicesSelectedClass.append(i)
        
        #print("Interpolated samples generated:", len(indicesSelectedClass))
        
        return Xint[indicesSelectedClass,:,:], np.repeat(selected_class,len(indicesSelectedClass))

    def fit(self, X, y):
        
        #print("In TrainVAE fit")
        
        X_c = X.copy()
        y_c = y.copy()
        
        if self.use_augmented_data:
            
            #check data
            classe_labels  = sorted(set(y))
            if classe_labels != [0,1]:
                raise Exception("Classe labels are not as expected 0 and 1!")
            
            for shift in [0]: #[0,5,8]
                for every_ith_sample in [10]: #[10,17]
                    X_interpolated1, y_interpolated1 = self._generateInterpolated(X_c, y_c, 1, shift, every_ith_sample)
                    
                    X_interpolated2, y_interpolated2 = self._generateInterpolated(X_c, y_c, 0, shift, every_ith_sample)
                    
                    X_c = np.concatenate((X_c, X_interpolated1), axis=0)
                    y_c = np.concatenate((y_c, y_interpolated1), axis=0)
                    
                    X_c = np.concatenate((X_c, X_interpolated2), axis=0)
                    y_c = np.concatenate((y_c, y_interpolated2), axis=0)
            
        #selected_class = 1 #the class number in y to be used
        iterations = self.vae_iterations #default 500
        hidden_layer_low = 100 #default 500 on CPU, 50 on GPU
        latent_dim = self.vae_latent_dim
        #onlyP300 = False #works with selected_class
        
        # if onlyP300 == True:
        
        #     selected_class_indices = np.where(y == selected_class)
            
        #     X = X[selected_class_indices,:,:]
        
        #     X = X[-1,:,:,:] #remove the first exta dimension
        
        #print("Samples count used by the VAE train: ", X.shape)
        
        #FIX: not the correct format (3360, 8, 257) but should be (3360, 257, 8)
        N, T, D = X_c.shape #N = number of samples, T = time steps, D = feature dimensions
        if T == D:
            print("WARNING: Your signal is a square matrix. Very, very unlikely unless you insist on using cov matrices instead of signal epochs.")
        print(N, T, D)
        # X = X.reshape(N,D,T)
        X_c = X_c.transpose(0,2,1)
        N, T, D = X_c.shape
        print(N, T, D)
        
        # min max scale the data  
        scaler = utils.MinMaxScaler()        
       
        scaled_data = scaler.fit_transform(X_c)
        
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
            batch_size = 4, #default 32
            epochs=iterations, #default 500
            shuffle = True,
            #callbacks=[early_stop_callback, reduceLR],
            verbose = 0,
        )
        
        self.modelVAE  = vae
        self.scalerVAE = scaler

        return self
    
    def _encode(self,X):
        
        print("Encode VAE")
        #timeVAE, scaler, X_train, X_test, y_train, y_test
        
        #Use the auto encoder to produce feature vectors
        
        # Process X_train
        X_c = X.copy()
        
        # convert to correct format expected by timeVAE
        N, T, D = X_c.shape #N = number of samples, T = time steps, D = feature dimensions
        if T == D:
            print("WARNING: Your signal is a square matrix. Very, very unlikely unless you insist on using cov matrices instead of signal epochs.")
        #print(N, T, D)
        X_c = X_c.transpose(0,2,1)
        N, T, D = X_c.shape
        #print(N, T, D)
        
        if self.use_pretrained_scaler:
            X_c_scaled = self.scalerVAE.transform(X_c)
        else:
            X_c_scaled = utils.MinMaxScaler().fit_transform(X_c)
        
        X_c_fv = self.modelVAE.encoder.predict(X_c_scaled,)
        
        X_c_fv_all = []
        
        if self.encoder_output == 3: #version that use all 3 outputs from the encoder
            for i in range(0, X.shape[0]):
                X_c_fv_all.append(np.concatenate((X_c_fv[0][i], X_c_fv[1][i], X_c_fv[2][i])))
        else:
            for i in range(0, X.shape[0]):
                X_c_fv_all.append(X_c_fv[self.encoder_output][i])
        
        X_c_fv_all = np.array(X_c_fv_all)  
        
        print("Are there are NaNs in the feature vector: ", np.isnan(X_c_fv_all).any())
        
        return X_c_fv_all #returns encoded X using the trained TimeVAE 

    def transform(self, X,):
        
        X_tansformed = self._encode(X)
        
        return X_tansformed

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.tansform(X)