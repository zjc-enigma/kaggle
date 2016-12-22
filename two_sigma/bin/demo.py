import numpy as np
import pandas as pd
import sqlite3
#from pandas.io import sql
#import h5py
#data_file = "../data/train.h5"
#hf = h5py.File(data_file, 'r') 
#print("keys in data:", list(hf.keys()))
#train_data = hf.get('train')
#column_name = list(train_data.keys())
#axis0 = train_data.get('axis0')
# axis1 = train_data.get('axis1')
# block0_items = train_data.get('block0_items')
# block0_values = train_data.get('block0_values')
# block1_items = train_data.get('block1_items')
# block1_values = train_data.get('block1_values')
# np_data = np.array(train_data)
# print("data.shape", np_data.shape)

#hdf = pd.read_hdf(data_file)

#sql.write_frame(hdf, name='two_sigma', con=conn)


def read_data_from_sqlite(conn, table_name):
    df = pd.read_sql("SELECT * from {table}".format(table=table_name), con=conn)
    return df


def write_data_to_sqlite(conn, df, table_name):
    df.to_sql(name=table_name, con=conn)



if __name__ == "__main__":
    conn = sqlite3.connect("../data/train.sqlite")
    table_name = "train"
    df = read_data_from_sqlite(conn, table_name)




    #write_data_to_sqlite(conn, df, table_name)


