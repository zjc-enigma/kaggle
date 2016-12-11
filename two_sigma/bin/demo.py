import numpy as np
import h5py

data_file = "../data/train.h5"


hf = h5py.File(data_file, 'r') 
print("keys in data:", list(hf.keys()))
train_data = hf.get('train')
np_data = np.array(train_data)
print("data.shape", np_data.shape)

