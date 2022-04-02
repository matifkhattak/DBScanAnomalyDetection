#########References##########
#https://donernesto.github.io/blog/outlier-detection-with-dbscan/
#https://donernesto.github.io/blog/outlier-detection-data-preparation/
#https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html
#https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/

# Standard library imports
from collections import Counter, defaultdict
import time
import os

# Third party imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (auc, average_precision_score,
                              roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.base import BaseEstimator

plt.rcParams.update({'font.size': 12})

# Misc. imports
#from blogutils import plot_metrics_curves, memotodisk

#import sklearn; print(sklearn.__version__)

np.random.seed(42)


#####1. Credit card dataset####
#See https://donernesto.github.io/blog/outlier-detection-data-preparation for details on this and other datasets

def loadData():
    with open(r"Dataset/anomaly_credit_Dataset.dat", "rb") as f:
        X_downsampled2_credit = np.load(f)
        y_downsampled2_credit = np.load(f)
    #    print('Total number of points downsampled2:{}. Number of positives: {} (fraction: {:.2%})'.format(
    #        len(y_downsampled2_credit), y_downsampled2_credit.sum(), y_downsampled2_credit.mean()))
    return X_downsampled2_credit,y_downsampled2_credit



X_downsampled2_credit,y_downsampled2_credit =loadData()
print("Origonal labels =", set(y_downsampled2_credit))
#exit(0)
###Example: single prediction###
#This looks promising: the positives are present mainly in the smaller classes and outlier class (by fraction). Let us visualize the results.
db = DBSCAN(eps=5, min_samples=10).fit(X_downsampled2_credit)
#print("Predicted cluster Labels=",set(db.labels_))
#print((y_downsampled2_credit==0).sum())
#print(np.sum(y_downsampled2_credit[db.labels_==0]))

for i in set(db.labels_):
    print("Value of i =", i)
    print('class {}: number of points in db.labels_ {:d}, number of positives {} (fraction: {:.3%})'.format(
        i,  np.sum(db.labels_==i), y_downsampled2_credit[db.labels_==i].sum(),
        y_downsampled2_credit[db.labels_==i].mean()))
#class 0: number of points 2766, number of positives 11 (fraction: 0.398%)
#class 1: number of points 15, number of positives 0 (fraction: 0.000%)
#class -1: number of points 312, number of positives 87 (fraction: 27.885%)
print('Total number of point: {}. Number of positives: {} (fraction: {:.2%})'.format(
            len(y_downsampled2_credit), y_downsampled2_credit.sum(), y_downsampled2_credit.mean()))
#Total number of point: 3093. Number of positives: 98 (fraction: 3.17%)


#####Visualizing the large dimensional data using TSNE=======
MAX_N_TSNE = 3500 #Avoid overly long computation times with TSNE. Values < 5000 recommended
neg = y_downsampled2_credit == 0
pos = y_downsampled2_credit == 1
assert len(X_downsampled2_credit) <= MAX_N_TSNE, 'Using a dataset with more than {} points is not recommended'.format(
                                            MAX_N_TSNE)
X_2D = TSNE(n_components=2, perplexity=30, n_iter=400).fit_transform(X_downsampled2_credit) # collapse in 2-D space for plotting


for i in set(db.labels_):
    print('class {}: number of points {:d}, fraction of positives {:.3%}'.format(i,
                                                                np.sum(db.labels_==i),
                                                                y_downsampled2_credit[db.labels_==i].mean()))
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#ax.scatter(X_2D[pos, 0], X_2D[pos, 1],color=['green'], marker='x', s=80, label='Positive') #c=[[0.8, 0.4, 0.4],]
count = 0
for i in set(db.labels_):
    if i == -1:
        #outlier according to dbscan
        ax.scatter(X_2D[db.labels_==i, 0], X_2D[db.labels_==i, 1], color=['red'], s=8, label='DBSCAN Outlier')
    else:
        if count==0:
            ax.scatter(X_2D[db.labels_==i, 0], X_2D[db.labels_==i, 1], color=['blue'], label='DBSCAN class {}'.format(i))
        else:
            ax.scatter(X_2D[db.labels_ == i, 0], X_2D[db.labels_ == i, 1], s=8, color=['magenta'],
                   label='DBSCAN class {}'.format(i))
        count = count+1

plt.axis('off')
plt.legend()
plt.show()

#Note how the outliers are indeed removed from the main cluster, and how the "outlier cluster" is correctly marked as a DBSCAN outlier class of -1. TSNE brings them altogether, altough they -according to DBSCAN- do not form a single cluster (the "-1" points don't belong anywhere). The actual outliers (crosses) are typically well-identified.