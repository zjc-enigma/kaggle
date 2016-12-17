import numpy as np
import pandas as pd
import h5py

data_file = "../data/train.h5"


hf = h5py.File(data_file, 'r') 
print("keys in data:", list(hf.keys()))
train_data = hf.get('train')
column_name = list(train_data.keys())
axis0 = train_data.get('axis0')
axis1 = train_data.get('axis1')
block0_items = train_data.get('block0_items')
block0_values = train_data.get('block0_values')
block1_items = train_data.get('block1_items')
block1_values = train_data.get('block1_values')
np_data = np.array(train_data)
print("data.shape", np_data.shape)

hdf = pd.read_hdf(data_file)
