# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:13:56 2022

@author: antona

pip install ydata-synthetic

An example of how TimeGan can be used to generate synthetic time-series data.
Dataset is Google's stock.
Input is int the form: (epochs, timesteps, features/channels), note that timesteps and features are inversed.

source: https://github.com/ydataai/ydata-synthetic/blob/dev/examples/timeseries/TimeGAN_Synthetic_stock_data.ipynb

"""

from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers import ModelParameters
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN

# fixes error: https://discuss.tensorflow.org/t/optimization-loop-failed-cancelled-operation-was-cancelled/1524/27
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#Specific to TimeGANs
seq_len=24  # Timesteps
n_seq = 6   # Features

hidden_dim=24
gamma=1

noise_dim = 32
dim = 128
batch_size = 128

log_step = 100
learning_rate = 5e-4

gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           noise_dim=noise_dim,
                           layers_dim=dim)

stock_data = processed_stock(path='GOOG.csv', seq_len=seq_len)
print("(epochs, timesteps, features/channels)",np.asarray(stock_data).shape)

# Training the TimeGAN synthetizer
if path.exists('synthesizer_stock.pkl'):
    synth = TimeGAN.load('synthesizer_stock.pkl')
else:
    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=seq_len, n_seq=n_seq, gamma=1)
    synth.train(stock_data, train_steps=50) #default train_steps = 50000
    synth.save('synthesizer_stock.pkl') #save trained model

#The generated synthetic stock data
synth_data = synth.sample(len(stock_data))
print(synth_data.shape)

cols = ['Open','High','Low','Close','Adj Close','Volume']

# #Plotting some generated samples. Both Synthetic and Original data are still standartized with values between [0,1]
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
# axes=axes.flatten()

# time = list(range(1,25))
# obs = np.random.randint(len(stock_data))

# for j, col in enumerate(cols):
#     df = pd.DataFrame({'Real': stock_data[obs][:, j],
#                    'Synthetic': synth_data[obs][:, j]})
#     df.plot(ax=axes[j],
#             title = col,
#             secondary_y='Synthetic data', style=['-', '--'])
# fig.tight_layout()

# #Evaluation of the generated synthetic data (PCA and TSNE)
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE

# sample_size = 250
# idx = np.random.permutation(len(stock_data))[:sample_size]

# real_sample = np.asarray(stock_data)[idx]
# synthetic_sample = np.asarray(synth_data)[idx]

# #for the purpose of comparision we need the data to be 2-Dimensional. For that reason we are going to use only two componentes for both the PCA and TSNE.
# synth_data_reduced = real_sample.reshape(-1, seq_len)
# stock_data_reduced = np.asarray(synthetic_sample).reshape(-1,seq_len)

# n_components = 2
# pca = PCA(n_components=n_components)
# tsne = TSNE(n_components=n_components, n_iter=300)

# #The fit of the methods must be done only using the real sequential data
# pca.fit(stock_data_reduced)

# pca_real = pd.DataFrame(pca.transform(stock_data_reduced))
# pca_synth = pd.DataFrame(pca.transform(synth_data_reduced))

# data_reduced = np.concatenate((stock_data_reduced, synth_data_reduced), axis=0)
# tsne_results = pd.DataFrame(tsne.fit_transform(data_reduced))

#Train synthetic test real (TSTR)
from tensorflow.keras import Input, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError

#First implement a simple RNN model for prediction
def RNN_regression(units):
    opt = Adam(name='AdamOpt')
    loss = MeanAbsoluteError(name='MAE')
    model = Sequential()
    model.add(GRU(units=units, #Gated Recurrent Unit
                  name=f'RNN_1'))
    model.add(Dense(units=6,
                    activation='sigmoid',
                    name='OUT'))
    model.compile(optimizer=opt, loss=loss)
    return model

#Prepare the dataset for the regression model
stock_data=np.asarray(stock_data)
synth_data = synth_data[:len(stock_data)]
n_events = len(stock_data)

#Split data on train and test
idx = np.arange(n_events)
n_train = int(.75*n_events)
train_idx = idx[:n_train]
test_idx = idx[n_train:]

#Define the X for synthetic and real data
X_stock_train = stock_data[train_idx, :seq_len-1, :]
X_synth_train = synth_data[train_idx, :seq_len-1, :]

X_stock_test = stock_data[test_idx, :seq_len-1, :]
y_stock_test = stock_data[test_idx, -1, :]

#Define the y for synthetic and real datasets
y_stock_train = stock_data[train_idx, -1, :]
y_synth_train = synth_data[train_idx, -1, :]

print('Synthetic X train: {}'.format(X_synth_train.shape))
print('Real X train: {}'.format(X_stock_train.shape))

print('Synthetic y train: {}'.format(y_synth_train.shape))
print('Real y train: {}'.format(y_stock_train.shape))

print('Real X test: {}'.format(X_stock_test.shape))
print('Real y test: {}'.format(y_stock_test.shape))

#Training the model with the real train data
ts_real = RNN_regression(12)
early_stopping = EarlyStopping(monitor='val_loss')

real_train = ts_real.fit(x=X_stock_train,
                          y=y_stock_train,
                          validation_data=(X_stock_test, y_stock_test),
                          epochs=200,
                          batch_size=128,
                          callbacks=[early_stopping])

#Training the model with the synthetic data
ts_synth = RNN_regression(12)
synth_train = ts_synth.fit(x=X_synth_train,
                          y=y_synth_train,
                          validation_data=(X_stock_test, y_stock_test),
                          epochs=200,
                          batch_size=128,
                          callbacks=[early_stopping])

#Summarize the metrics here as a pandas dataframe
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error
real_predictions = ts_real.predict(X_stock_test)
synth_predictions = ts_synth.predict(X_stock_test)

metrics_dict = {'r2': [r2_score(y_stock_test, real_predictions),
                       r2_score(y_stock_test, synth_predictions)],
                'MAE': [mean_absolute_error(y_stock_test, real_predictions),
                        mean_absolute_error(y_stock_test, synth_predictions)],
                'MRLE': [mean_squared_log_error(y_stock_test, real_predictions),
                         mean_squared_log_error(y_stock_test, synth_predictions)]}

results = pd.DataFrame(metrics_dict, index=['Real', 'Synthetic'])

print(results)