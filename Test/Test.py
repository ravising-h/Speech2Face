
import os
import pandas as pd
import librosa
import numpy as np
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D,AveragePooling2D
from keras.models import Sequential, Model
from keras.regularizers import l2

def adjust(stft):
  if stft.shape[1] == 601:
    return stft
  else:
    return np.concatenate((stft,stft[:,0:601 - stft.shape[1]]),axis = 1)

file_path = XXXXXXXXXXXXXXXXXXXXXXXXX
X = np.zeros((1,257,601,2),dtype = np.float32)
wav_file , sr = librosa.load(file_path,sr = 16000, duration = 6.0 ,mono = True)
    
stft_ = librosa.core.stft(wav_file, n_fft = 512, hop_length = int(np.ceil(0.01 * sr)),win_length = int(np.ceil(0.025 * sr)) , window='hann', center=True,pad_mode='reflect')

stft = adjust(stft_)
for j in range(stft.shape[0]):
  for k in range(stft.shape[1]):
    X[0,j,k,0] = stft[j,k].real
    X[0,j,k,1] = stft[j,k].imag

X = np.sign(X) * ( np.abs(X) ** 0.3 )
#np.save("X.npy",X)

from keras.optimizers import Adam
from keras import losses
adam = Adam(lr=0.001, beta_1=0.5, beta_2=0.9799, amsgrad=False)

model = Sequential()
model.add(Conv2D(70,  kernel_size = 4, strides=1, activation='relu' ,input_shape = (X.shape[1],X.shape[2],2)))
model.add(BatchNormalization())
model.add(Conv2D(70,  kernel_size = 4, strides=1, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(140, kernel_size = 4, strides=1, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,1),strides = (2,1)))
model.add(Conv2D(140, kernel_size = 4, strides=1, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,1),strides = (2,1)))
model.add(Conv2D(140, kernel_size = 3, strides=1, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,1),strides = (2,1)))
model.add(Conv2D(275, kernel_size = 4, strides=1, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,1),strides = (2,1)))
model.add(Conv2D(530, kernel_size = 4, strides=1, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(530, kernel_size = 4,  strides=2,activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Conv2D(530, kernel_size = 4, strides=2, activation='relu'))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(1,1), strides=3, padding='valid'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation="relu")) 
model.add(Dense(4096, activation="relu")) 

# COMPILE WITH ADAM OPTIMIZER AND CROSS ENTROPY COST
model.compile(optimizer=adam, loss="mean_absolute_error", metrics=["mse"])

model.load_weights("weights.h5")

y = model.predict(X)
print(y)
