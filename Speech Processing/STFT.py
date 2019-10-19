##########################
# Importing Libraries
##########################

import os
import pandas as pd
import librosa
import numpy as np
import random
from tqdm import tqdm


##########################
# Running Wav file In loop
##########################

def adjust(stft):
  if stft.shape[1] == 601:
    return stft
  else:
    return np.concatenate((stft,stft[:,0:601 - stft.shape[1]]),axis = 1)

################################
# Extracting STFT of Utterances
################################

metadata = pd.read_csv("/content/drive/My Drive/datasetss/vox1_meta.csv", delimiter = "\t")
os.chdir("/content/wav")
folder_name = os.listdir()
no_of_records = 3
X = np.zeros((len(folder_name) *  no_of_records,257,601,2),dtype = np.float32)
index = 0 
for folder in tqdm(folder_name):
  for i in range(no_of_records):
    
    path_ = os.path.join(folder,random.choice(os.listdir(folder)))
    file_path = os.path.join(path_,random.choice(os.listdir(path_)))
    wav_file , sr = librosa.load(file_path,sr = 16000, duration = 6.0 ,mono = True)
    
    stft_ = librosa.core.stft(wav_file, n_fft = 512, hop_length = int(np.ceil(0.01 * sr)),win_length = int(np.ceil(0.025 * sr)) , window='hann', center=True,pad_mode='reflect')
    
    stft = adjust(stft_)
    for j in range(stft.shape[0]):
      for k in range(stft.shape[1]):
        X[index,j,k,0] = stft[j,k].real
        X[index,j,k,1] = stft[j,k].imag

    index += 1


###################################
# Power Law Compression
###################################

X = np.sign(X) * ( np.abs(X) ** 0.3 )

####################################
# Saving File
####################################

np.save("/content/drive/My Drive/wavX.npy",X)
