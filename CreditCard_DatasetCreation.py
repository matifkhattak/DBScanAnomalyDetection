#########References##########
#https://donernesto.github.io/blog/outlier-detection-with-dbscan/
#https://donernesto.github.io/blog/outlier-detection-data-preparation/
#https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html
#https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/

# Standard library imports
import time
import os

# Third party library imports
import scipy.io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

plt.rcParams.update({'font.size': 12})

def downsample_scale_split_df(df_full, y_column='Class', frac_negative=1, frac_positive=1, scaler=RobustScaler,
                        random_state=1, verbose=False):
    """ Returns downsampled X, y DataFrames, with prescribed downsampling of positives and negatives
    The labels (y's) should have values 0, 1 and be located in y_column
    X will additionally be scaled using the passed scaler

    Arguments
    =========
    df_full (pd.DataFrame) : data to be processed
    y_column (str) : name of the column containing the Class
    frac_negative (int): fraction of negatives in returned data
    frac_positive (int): fraction of negatives in returned data
    scaler (sci-kit learn scaler object)

    Returns
    ========
    downsampled and scaled X (DataFrame) and downsampled y (Series)
    """
    df_downsampled = (pd.concat([df_full.loc[df_full[y_column] == 0].sample(frac=frac_negative,
                                                                        random_state=random_state),
                                df_full.loc[df_full[y_column] == 1].sample(frac=frac_positive,
                                                                       random_state=random_state)])
                              .sample(frac=1, random_state=random_state)) # a random shuffle to mix both classes
    X_downsampled = df_downsampled.loc[:, df_full.columns != y_column]
    y_downsampled = df_downsampled.loc[:, y_column]
    if scaler is not None:
        X_downsampled = scaler().fit_transform(X_downsampled) # Scale the data
    if verbose:
        print('Number of points: {}, number of positives: {} ({:.2%})'.format(
            len(y_downsampled), y_downsampled.sum(), y_downsampled.mean()))
    return(X_downsampled, y_downsampled)

#####1. Credit Card dataset#####
df_full = pd.read_csv(r"Dataset/creditcard_Orignal.csv")
df_full = df_full.drop('Time', axis=1)
df_full = df_full.sample(frac=1) # Shuffle the data set
df_full.shape
num_neg = (df_full.Class==0).sum()
num_pos = df_full.Class.sum()
print('Number of positive / negative samples: {} / {}'.format(num_pos, num_neg))
print('Fraction of positives: {:.2%}'.format(num_pos/num_neg))

print(df_full.columns)
print(df_full.shape)

#Minimal feature transformation (before general scaling) will be done; only the Amount feature will be log-transformed. Adding 0.01 results in a nice, symmetric distribution.
df_full.Amount = np.log(0.01 + df_full.Amount)
df_full.Amount.plot(kind='box')
plt.show()

# Looking at the boxplots of the first 10 features, the features are characterized by strong outliers (large deviation from the median, normalized by the IQR):

df_scaled = pd.concat((pd.DataFrame(RobustScaler().fit_transform(df_full.iloc[:, :-1])), df_full.Class), axis=1)
df_scaled.columns = df_full.columns
range_1 = np.r_[1:10, 29]
df_long = pd.melt(df_scaled.iloc[:, range_1],
                  "Class", var_name="feature", value_name="value")
sns.factorplot("feature", hue="Class", y="value", data=df_long, kind="box", size=5, aspect=2)
plt.ylim([-20, 20])
plt.show()

#We will create the datasets for further usage. There will be one downsample dataset (strongly downsampled: only 10% of positives), because of computational limitations of some algorithms.

X_credit, y_credit = downsample_scale_split_df(df_full, verbose=1, random_state=1)
#X_downsampled_credit, y_downsampled_credit = downsample_scale_split_df(df_full, frac_positive=0.2,
#                                                           frac_negative=0.1, verbose=1, random_state=1,
#                                                            scaler=RobustScaler)

X_downsampled2_credit, y_downsampled2_credit = downsample_scale_split_df(df_full, frac_positive=0.2,
                                                            frac_negative=3000/len(y_credit), verbose=1, random_state=1,
                                                            scaler=RobustScaler)

with open(r"Dataset/anomaly_credit_Dataset.dat","wb") as f:
    np.save(f, X_downsampled2_credit)
    np.save(f, y_downsampled2_credit)