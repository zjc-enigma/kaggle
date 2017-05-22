import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
enc = OneHotEncoder()

train_path = '../data/train.csv'
test_path = '../data/test.csv'




def read(fp):
    df = (pd.read_csv(fp)
          .drop(['id', 'timestamp'], axis=1)
          .fillna(0)
          .pipe(make_label)
          .pipe(make_one_hot)
    )

    return df


#train_df = pd.read_csv(train_path)
#test_df = pd.read_csv(test_path)

#train_df.info()
# Columns: 292 entries, id to price_doc
# dtypes: float64(119), int64(157), object(16)

# distribution of numerical features
#train_df.describe()


# distribution of categorical features
#train_df.describe(include=['O'])


def make_label(df):
    object_df = df.select_dtypes(include=[object])
    numeric_df = df.select_dtypes(exclude=[object])
    labeled_df = object_df.apply(lenc.fit_transform)
    df = pd.concat([numeric_df, labeled_df], 1)
    return df



def make_one_hot(df):

    object_df = df.select_dtypes(include=[object])
    numeric_df = df.select_dtypes(exclude=[object])
    labeled_df = object_df.apply(enc.fit_transform)
    df = pd.concat([numeric_df, labeled_df], 1)
    return df





# y_train = train_df['price_doc']
# x_train = train_df.drop(['price_doc', 'id', 'timestamp'], axis=1)
# x_train = x_train.fillna(0)
# object_df = x_train.select_dtypes(include=[object])
# numeric_df = x_train.select_dtypes(exclude=[object])
# labeled_df = object_df.apply(lenc.fit_transform)
# x_train = pd.concat([numeric_df, labeled_df], 1)
# x_train = enc.fit_transform(x_train)


train_df = read(train_path)
t_df = pd.read_csv(train_path)
t_df = t_df.drop(['id', 'timestamp'], axis=1)
t_df = t_df.fillna(0)

object_df = t_df.select_dtypes(include=[object])
numeric_df = t_df.select_dtypes(exclude=[object])
object_df.apply(lenc.fit_transform)

# pd.concat([numeric_df, labeled_df], 1)
