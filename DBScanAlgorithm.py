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
import sklearn
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

def buildDBScanModel(rfm, rfm_df_scaled,MRType):

    ###Example: single prediction###
    #This looks promising: the positives are present mainly in the smaller classes and outlier class (by fraction). Let us visualize the results.
    dbScanModel = DBSCAN(eps=5, min_samples=10).fit(rfm_df_scaled)
    # Assign cluster labels
    rfm['Cluster_Id'] = dbScanModel.labels_
    return rfm, rfm_df_scaled, dbScanModel

def dbScanAlgorithm(MRType = "", rfm_WithClassLabels = None, rowIndex = None, valueToIncrement = None):

    scaler = StandardScaler()# Instantiate

    X_credit, y_credit = loadData()
    X_credit = pd.DataFrame(X_credit)
    X_credit_scaled = scaler.fit_transform(X_credit)
    X_credit_scaled= pd.DataFrame(X_credit_scaled)
    # print("Origonal labels =", set(y_downsampled2_credit))
    # exit(0)

    if (MRType == "MR1"):
        print("MR1-Followup")

        # MR1.1: adding single point
        # # For adding an instance having an index in any of [183, 267, 618, 767, 829, 874, 912, 1437, 1513, 1919, 1960, 2114, 2179, 2247, 2314, 2411, 2414, 2432, 2436, 2437, 2467, 2635, 2641, 2722, 2942, 2967, 2994]. we got inconsistent results
        #duplicateRow = X_credit_scaled.iloc[183] # X_credit_scaled.iloc[rowIndex]
        #X_credit = X_credit.append(duplicateRow, ignore_index=True)
        #X_credit_scaled = X_credit_scaled.append(duplicateRow, ignore_index=True)

        # MR1.2 adding multiple points (each belonging to different cluster).
        #duplicateRow = X_credit_scaled.iloc[183]  # index183 datapoint belong to cluter -1 (noise)
        #X_credit = X_credit.append(duplicateRow, ignore_index=True)
        #X_credit_scaled = X_credit_scaled.append(duplicateRow, ignore_index=True)
        #duplicateRow = X_credit_scaled.iloc[267]  # index267 datapoint belong to cluster0
        #X_credit = X_credit.append(duplicateRow, ignore_index=True)
        #X_credit_scaled = X_credit_scaled.append(duplicateRow, ignore_index=True)
        #duplicateRow = X_credit_scaled.iloc[2994] # index2994 datapoint belong to cluster1
        #X_credit = X_credit.append(duplicateRow, ignore_index=True)
        #X_credit_scaled = X_credit_scaled.append(duplicateRow, ignore_index=True)

        # MR1.3 not valid for DBScan clustering.
    if (MRType == "MR2"):
        print("MR2-Followup")
        X_credit_scaled = scaler.fit_transform(X_credit)

    if (MRType == "MR3"):
        print("MR3-Followup")
        #print(X_credit_scaled.iloc[:,28])
        #exit(0)
        X_credit_scaled["0_New"] = X_credit_scaled.iloc[:,0]
        X_credit_scaled["1_New"] = X_credit_scaled.iloc[:,1]
        X_credit_scaled["2_New"] = X_credit_scaled.iloc[:,2]
        X_credit_scaled["3_New"] = X_credit_scaled.iloc[:, 3]
        X_credit_scaled["4_New"] = X_credit_scaled.iloc[:, 4]
        X_credit_scaled["5_New"] = X_credit_scaled.iloc[:, 5]
        X_credit_scaled["6_New"] = X_credit_scaled.iloc[:, 6]
        X_credit_scaled["7_New"] = X_credit_scaled.iloc[:, 7]
        X_credit_scaled["8_New"] = X_credit_scaled.iloc[:, 8]
        X_credit_scaled["9_New"] = X_credit_scaled.iloc[:, 9]
        X_credit_scaled["10_New"] = X_credit_scaled.iloc[:, 10]
        X_credit_scaled["11_New"] = X_credit_scaled.iloc[:, 11]
        X_credit_scaled["12_New"] = X_credit_scaled.iloc[:, 12]
        X_credit_scaled["13_New"] = X_credit_scaled.iloc[:, 13]
        X_credit_scaled["14_New"] = X_credit_scaled.iloc[:, 14]
        X_credit_scaled["15_New"] = X_credit_scaled.iloc[:, 15]
        X_credit_scaled["16_New"] = X_credit_scaled.iloc[:, 16]
        X_credit_scaled["17_New"] = X_credit_scaled.iloc[:, 17]
        X_credit_scaled["18_New"] = X_credit_scaled.iloc[:, 18]
        X_credit_scaled["19_New"] = X_credit_scaled.iloc[:, 19]
        X_credit_scaled["20_New"] = X_credit_scaled.iloc[:, 20]
        X_credit_scaled["21_New"] = X_credit_scaled.iloc[:, 21]
        X_credit_scaled["22_New"] = X_credit_scaled.iloc[:, 22]
        X_credit_scaled["23_New"] = X_credit_scaled.iloc[:, 23]
        X_credit_scaled["24_New"] = X_credit_scaled.iloc[:, 24]
        X_credit_scaled["25_New"] = X_credit_scaled.iloc[:, 25]
        X_credit_scaled["26_New"] = X_credit_scaled.iloc[:, 26]
        X_credit_scaled["27_New"] = X_credit_scaled.iloc[:, 27]
        X_credit_scaled["28_New"] = X_credit_scaled.iloc[:, 28]

    elif (MRType == "MR4"):
        print("MR4-Followup")

        # When we duplicate the intance(s) at these indexes [13, 234, 486, 495, 504, 881, 957, 1201, 1253, 1303, 1437, 1552, 1619, 1662, 2179, 2246, 2352, 2413, 2547, 2548, 2649, 2700, 2764, 2833, 2922], we got inconsistent output for the follow-up input
        #     , we got inconsistent outputs
        ## (i) removing single point
        X_credit.drop(13, inplace=True) #rowIndex is used to first identify the indexes for which the model gives inconsistent results for both source and follow-up execution
        X_credit_scaled.drop(13,inplace=True)

        ## (ii) removing multiple points (each belonging to different cluster i.e., from cluster0 and cluster2)
        #X_credit.drop(13, inplace=True)  # For source execution, data instance at index 13 belongs to cluster0
        #X_credit_scaled.drop(13, inplace=True)
        #X_credit.drop(1437, inplace=True)  # For source execution, data instance at index 1437 belongs to cluster1
        #X_credit_scaled.drop(1437, inplace=True)
        #X_credit.drop(3069, inplace=True)  # For source execution, data instance at index 3069 belongs to cluster-1
        #X_credit_scaled.drop(3069, inplace=True)

        # (iii) Removing 1000 rows belonging to cluster_Id = 0
        #newDF = rfm_WithClassLabels.loc[rfm_WithClassLabels['Cluster_Id'] == 0]
        ## print(newDF.head(1000).index)
        #X_credit = X_credit.drop(newDF.head(1000).index)
        #X_credit_scaled = X_credit_scaled.drop(newDF.head(1000).index)
        ## print(rfm_WithClassLabels)
        ## exit(0)
    elif (MRType == "MR5"):
        print("MR5-Followup")
        X_credit['NewInformativeAttribute'] = 0# Add a new uninformative attribute (any constant) to all the instances
        X_credit_scaled['NewInformativeAttribute'] = 0  # Add a new uninformative attribute (any constant) to all the instances
        # OR X_credit_scaled['NewInformativeAttribute'] = 69 # Add a new uninformative attribute (any constant) to all the instances

    elif (MRType == "MR6" or MRType == "MR6Followup"):
        print("MR6-Source")
        # For MR3, we found that if we add duplicate instance(s) having an index 13, 234, we got inconsistent results, so we choose those two points to add a new point in between them
        divisor = 4
        col0Value = (float(rfm_Source.iloc[13:14, 0]) + float(rfm_Source.iloc[234:235, 0])) / divisor
        col1Value = (float(rfm_Source.iloc[13:14, 1]) + float(rfm_Source.iloc[234:235, 1])) / divisor
        col2Value = (float(rfm_Source.iloc[13:14, 2]) + float(rfm_Source.iloc[234:235, 2])) / divisor
        col3Value = (float(rfm_Source.iloc[13:14, 3]) + float(rfm_Source.iloc[234:235, 3])) / divisor
        col4Value = (float(rfm_Source.iloc[13:14, 4]) + float(rfm_Source.iloc[234:235, 4])) / divisor
        col5Value = (float(rfm_Source.iloc[13:14, 5]) + float(rfm_Source.iloc[234:235, 5])) / divisor
        col6Value = (float(rfm_Source.iloc[13:14, 6]) + float(rfm_Source.iloc[234:235, 6])) / divisor
        col7Value = (float(rfm_Source.iloc[13:14, 7]) + float(rfm_Source.iloc[234:235, 7])) / divisor
        col8Value = (float(rfm_Source.iloc[13:14, 8]) + float(rfm_Source.iloc[234:235, 8])) / divisor
        col9Value = (float(rfm_Source.iloc[13:14, 9]) + float(rfm_Source.iloc[234:235, 9])) / divisor
        col10Value = (float(rfm_Source.iloc[13:14, 10]) + float(rfm_Source.iloc[234:235, 10])) / divisor
        col11Value = (float(rfm_Source.iloc[13:14, 11]) + float(rfm_Source.iloc[234:235, 11])) / divisor
        col12Value = (float(rfm_Source.iloc[13:14, 12]) + float(rfm_Source.iloc[234:235, 12])) / divisor
        col13Value = (float(rfm_Source.iloc[13:14, 13]) + float(rfm_Source.iloc[234:235, 13])) / divisor
        col14Value = (float(rfm_Source.iloc[13:14, 14]) + float(rfm_Source.iloc[234:235, 14])) / divisor
        col15Value = (float(rfm_Source.iloc[13:14, 15]) + float(rfm_Source.iloc[234:235, 15])) / divisor
        col16Value = (float(rfm_Source.iloc[13:14, 16]) + float(rfm_Source.iloc[234:235, 16])) / divisor
        col17Value = (float(rfm_Source.iloc[13:14, 17]) + float(rfm_Source.iloc[234:235, 17])) / divisor
        col18Value = (float(rfm_Source.iloc[13:14, 18]) + float(rfm_Source.iloc[234:235, 18])) / divisor
        col19Value = (float(rfm_Source.iloc[13:14, 19]) + float(rfm_Source.iloc[234:235, 19])) / divisor
        col20Value = (float(rfm_Source.iloc[13:14, 20]) + float(rfm_Source.iloc[234:235, 20])) / divisor
        col21Value = (float(rfm_Source.iloc[13:14, 21]) + float(rfm_Source.iloc[234:235, 21])) / divisor
        col22Value = (float(rfm_Source.iloc[13:14, 22]) + float(rfm_Source.iloc[234:235, 22])) / divisor
        col23Value = (float(rfm_Source.iloc[13:14, 23]) + float(rfm_Source.iloc[234:235, 23])) / divisor
        col24Value = (float(rfm_Source.iloc[13:14, 24]) + float(rfm_Source.iloc[234:235, 24])) / divisor
        col25Value = (float(rfm_Source.iloc[13:14, 25]) + float(rfm_Source.iloc[234:235, 25])) / divisor
        col26Value = (float(rfm_Source.iloc[13:14, 26]) + float(rfm_Source.iloc[234:235, 26])) / divisor
        col27Value = (float(rfm_Source.iloc[13:14, 27]) + float(rfm_Source.iloc[234:235, 27])) / divisor
        col28Value = (float(rfm_Source.iloc[13:14, 28]) + float(rfm_Source.iloc[234:235, 28])) / divisor

        # Treated as source: time1
        X_credit.loc[len(X_credit.index)] = [col0Value, col1Value, col2Value, col3Value, col4Value, col5Value,
                                             col6Value, col7Value, col8Value, col9Value, col10Value, col11Value,
                                             col12Value, col13Value, col14Value, col15Value, col16Value, col17Value,
                                             col18Value, col19Value, col20Value, col21Value, col22Value, col23Value,
                                             col24Value, col25Value, col26Value, col27Value, col28Value]
        X_credit_scaled.loc[len(X_credit_scaled.index)] = [col0Value, col1Value, col2Value, col3Value, col4Value,
                                                           col5Value,
                                                           col6Value, col7Value, col8Value, col9Value, col10Value,
                                                           col11Value,
                                                           col12Value, col13Value, col14Value, col15Value, col16Value,
                                                           col17Value,
                                                           col18Value, col19Value, col20Value, col21Value, col22Value,
                                                           col23Value,
                                                           col24Value, col25Value, col26Value, col27Value, col28Value]

        if (MRType == "MR6Followup"):
            print("MR6-Followup")
            # time2: In next step, now shuffle the data-points, treated as follow-up: the result for this new instance + other instances should remain consistent
            # Note: When executing time2 code, don't comment out the time1 execution code. For follow-up execution, both the time1 and time2 code should be uncommented and executed.
            X_credit = sklearn.utils.shuffle(X_credit, random_state=1)
            X_credit_scaled = sklearn.utils.shuffle(X_credit_scaled, random_state=1)
    elif (MRType == "MR7"):
        print("MR7-Followup")
        print("ValueToIncrement = ", valueToIncrement)
        X_credit += valueToIncrement
        X_credit_scaled += valueToIncrement

    elif (MRType == "MR8"):
        print("MR8-Followup")
        print("ValueToBeMultiplied = ", valueToIncrement)
        X_credit *= valueToIncrement
        X_credit_scaled *= valueToIncrement
    elif (MRType == "MR9"):
        print("MR9-Followup")
        # For MR3, we found that if we add duplicate instance(s) having an index 13, 234, we get inconsistent results, so we choose those two points to add a new point in mid of them


        # MR9.1: replace one instance of cluster 0#####
        #X_credit.loc[486] = X_credit.loc[234]  # instances at index 234 and 486 belongs to same cluster i.e. cluster0
        #X_credit_scaled.loc[486] = X_credit_scaled.loc[234]

        #### MR9.2: replace multiple instances of cluster 0#####
        X_credit.loc[13] = X_credit.loc[234]  # instances at index13, 234, 486,504,881 belongs to same cluster i.e. cluster0
        X_credit_scaled.loc[13] = X_credit_scaled.loc[234]
        X_credit.loc[486] = X_credit.loc[234]
        X_credit_scaled.loc[486] = X_credit_scaled.loc[234]
        X_credit.loc[495] = X_credit.loc[234]
        X_credit_scaled.loc[495] = X_credit_scaled.loc[234]
        X_credit.loc[504] = X_credit.loc[234]
        X_credit_scaled.loc[504] = X_credit_scaled.loc[234]
        X_credit.loc[881] = X_credit.loc[234]
        X_credit_scaled.loc[881] = X_credit_scaled.loc[234]


        #####MR9.3 replace all instances of cluster 0#####
        # print("MR9.3-Followup")
        #dfCluster0 = rfm_WithClassLabels[rfm_WithClassLabels['Cluster_Id'] == -1]
        #for rowIndex,row in dfCluster0.iterrows():
        #    X_credit.iloc[rowIndex,:] = X_credit_scaled.loc[3069:3069]
        #    X_credit_scaled.iloc[rowIndex,:] = X_credit_scaled.loc[3069:3069]

    elif (MRType == "MR10"):
        print("MR10-Followup")  # Perform swapping of features
        X_credit = X_credit[[28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]
        X_credit_scaled = X_credit_scaled[[28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0]]

    elif (MRType == "MR11"):
        print("MR11-Followup")

        X_credit['NewInformativeAttribute'] = 0
        X_credit['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 0.3  # the indexes assigned to rfm and rfm_WithClassLabels are not properly odered but the same indexes are assigned to instances to both dataframes, so this is the reason that the filter is applied on rfm_WithClassLabels not the dfWithProperIndexes
        X_credit['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 1] = 0.6
        X_credit['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == -1] = 0.9
        X_credit_scaled['NewInformativeAttribute'] = 0
        X_credit_scaled['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 0] = 0.3
        X_credit_scaled['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == 1] = 0.6
        X_credit_scaled['NewInformativeAttribute'].loc[rfm_WithClassLabels['Cluster_Id'] == -1] = 0.9

    elif (MRType == "MR12"):
        print("MR12-Followup")
        ##MR12.1: Reversing the data points##
        #X_credit = X_credit.loc[::-1]
        #X_credit_scaled = X_credit_scaled.loc[::-1]

        ##MR12.2: Random shuffling of data points
        X_credit = sklearn.utils.shuffle(X_credit, random_state=1)
        X_credit_scaled = sklearn.utils.shuffle(X_credit_scaled, random_state=1)

    elif (MRType == "MR13"):
        print("MR13-Followup")
        X_credit *= -1
        X_credit_scaled *= -1

    elif (MRType == "MR14"):
        print("MR14-Followup")

        # MR14.2 adding new data point(s) with uninformative attributes should not change the output (all features have 0 value).
        #for i in range(100):
        #    X_credit.loc[len(X_credit.index)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #    X_credit_scaled.loc[len(X_credit_scaled.index)] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        # Element at index0 (-0.723738  -0.752888  2.301611) and index1(1.731617   1.042467 -0.906466) both belong to cluster#2. Let add multiple data points inbetween these two data points
        # MR14.1 Add multiple points in between these two points to make this cluster more compact/dense

        for x in range(3, 54):
            col0Value = (float(rfm_Source.iloc[13:14, 0]) + float(rfm_Source.iloc[234:235, 0])) / 3
            col1Value = (float(rfm_Source.iloc[13:14, 1]) + float(rfm_Source.iloc[234:235, 1])) / 3
            col2Value = (float(rfm_Source.iloc[13:14, 2]) + float(rfm_Source.iloc[234:235, 2])) / 3
            col3Value = (float(rfm_Source.iloc[13:14, 3]) + float(rfm_Source.iloc[234:235, 3])) / 3
            col4Value = (float(rfm_Source.iloc[13:14, 4]) + float(rfm_Source.iloc[234:235, 4])) / 3
            col5Value = (float(rfm_Source.iloc[13:14, 5]) + float(rfm_Source.iloc[234:235, 5])) / 3
            col6Value = (float(rfm_Source.iloc[13:14, 6]) + float(rfm_Source.iloc[234:235, 6])) / 3
            col7Value = (float(rfm_Source.iloc[13:14, 7]) + float(rfm_Source.iloc[234:235, 7])) / 3
            col8Value = (float(rfm_Source.iloc[13:14, 8]) + float(rfm_Source.iloc[234:235, 8])) / 3
            col9Value = (float(rfm_Source.iloc[13:14, 9]) + float(rfm_Source.iloc[234:235, 9])) / 3
            col10Value = (float(rfm_Source.iloc[13:14, 10]) + float(rfm_Source.iloc[234:235, 10])) / 3
            col11Value = (float(rfm_Source.iloc[13:14, 11]) + float(rfm_Source.iloc[234:235, 11])) / 3
            col12Value = (float(rfm_Source.iloc[13:14, 12]) + float(rfm_Source.iloc[234:235, 12])) / 3
            col13Value = (float(rfm_Source.iloc[13:14, 13]) + float(rfm_Source.iloc[234:235, 13])) / 3
            col14Value = (float(rfm_Source.iloc[13:14, 14]) + float(rfm_Source.iloc[234:235, 14])) / 3
            col15Value = (float(rfm_Source.iloc[13:14, 15]) + float(rfm_Source.iloc[234:235, 15])) / 3
            col16Value = (float(rfm_Source.iloc[13:14, 16]) + float(rfm_Source.iloc[234:235, 16])) / 3
            col17Value = (float(rfm_Source.iloc[13:14, 17]) + float(rfm_Source.iloc[234:235, 17])) / 3
            col18Value = (float(rfm_Source.iloc[13:14, 18]) + float(rfm_Source.iloc[234:235, 18])) / 3
            col19Value = (float(rfm_Source.iloc[13:14, 19]) + float(rfm_Source.iloc[234:235, 19])) / 3
            col20Value = (float(rfm_Source.iloc[13:14, 20]) + float(rfm_Source.iloc[234:235, 20])) / 3
            col21Value = (float(rfm_Source.iloc[13:14, 21]) + float(rfm_Source.iloc[234:235, 21])) / 3
            col22Value = (float(rfm_Source.iloc[13:14, 22]) + float(rfm_Source.iloc[234:235, 22])) / 3
            col23Value = (float(rfm_Source.iloc[13:14, 23]) + float(rfm_Source.iloc[234:235, 23])) / 3
            col24Value = (float(rfm_Source.iloc[13:14, 24]) + float(rfm_Source.iloc[234:235, 24])) / 3
            col25Value = (float(rfm_Source.iloc[13:14, 25]) + float(rfm_Source.iloc[234:235, 25])) / 3
            col26Value = (float(rfm_Source.iloc[13:14, 26]) + float(rfm_Source.iloc[234:235, 26])) / 3
            col27Value = (float(rfm_Source.iloc[13:14, 27]) + float(rfm_Source.iloc[234:235, 27])) / 3
            col28Value = (float(rfm_Source.iloc[13:14, 28]) + float(rfm_Source.iloc[234:235, 28])) / 3

            # Treated as source: time1
            X_credit.loc[len(X_credit.index)] = [col0Value, col1Value, col2Value, col3Value, col4Value, col5Value,
                                                 col6Value, col7Value, col8Value, col9Value, col10Value, col11Value,
                                                 col12Value, col13Value, col14Value, col15Value, col16Value, col17Value,
                                                 col18Value, col19Value, col20Value, col21Value, col22Value, col23Value,
                                                 col24Value, col25Value, col26Value, col27Value, col28Value]
            X_credit_scaled.loc[len(X_credit_scaled.index)] = [col0Value, col1Value, col2Value, col3Value, col4Value,
                                                               col5Value,
                                                               col6Value, col7Value, col8Value, col9Value, col10Value,
                                                               col11Value,
                                                               col12Value, col13Value, col14Value, col15Value, col16Value,
                                                               col17Value,
                                                               col18Value, col19Value, col20Value, col21Value, col22Value,
                                                               col23Value,
                                                               col24Value, col25Value, col26Value, col27Value, col28Value]


    rfm, rfm_df_scaled, clusterModel = buildDBScanModel(X_credit,X_credit_scaled,MRType)
    return rfm, rfm_df_scaled, clusterModel
if __name__ == '__main__':

    ##########################################################
    # ============Metamorphic Relations (MRs)=================#
    ##########################################################

    ###############################
    #####Source Executions======
    ###############################
    #print("#####Source Execution#####")
    rfm_Source, rfm_df_scaled_Source, dbScanModel = dbScanAlgorithm()


    # =========MR#1: Duplicating single, and multiple instances (each belonging to different class) =============#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    ##rfm_Source = pd.DataFrame(rfm_Source).reset_index()
    #conflictDataInstances = []
    #print(rfm_Source[rfm_Source["Cluster_Id"]==1])
    #exit(0)
    # Begin loop: This loop can be used to iterate over all instances to see for which instance (if we duplicate) we will get inconsistent results for the follow-up inputs
    #for rowIndex in range(3093):
    #    rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR1", rfm_Source,rowIndex)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    if (len(inconsistentOutDataRows)>0):
    #        conflictDataInstances.append(rowIndex)
    #        print("For instance at this index, we got inconsistent outputs = ", rowIndex)
    #print("Instances indexes for which we got different outputs = ", conflictDataInstances)
    #End loop

    # For adding an instance having an index in any of [183, 267, 618, 767, 829, 874, 912, 1437, 1513, 1919, 1960, 2114, 2179, 2247, 2314, 2411, 2414, 2432, 2436, 2437, 2467, 2635, 2641, 2722, 2942, 2967, 2994]. we got inconsistent results
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR1", rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)

    # =========MR#2: Apply normalization on the normalizeed data =============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR2")
    ##Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())

    # =========MR#3: Addition of features by copying/duplicating the original feature set=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR3")
    ## Merge (using inner join based on the indexes) the source(having all datapoints) with follow-up(with the left data), the common data points should be allocated to same clusters for both the source and follow-up executions
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("Inconsistent Data Rows at specific index =", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source.head(6))
    #print(rfm_Followup.head(6))

    # =========MR#4: Removing one, more instances (each belonging to different cluster) should not change the output =============#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    conflictDataInstances = []
    #for rowIndex in range(3093):
    #    rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR4", rfm_Source,rowIndex)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    if (len(inconsistentOutDataRows)>0):
    #        conflictDataInstances.append(rowIndex)
    #        print("For instance at this index, we got inconsistent outputs = ", rowIndex)
    #print("Instances indexes for which we got different outputs = ", conflictDataInstances)
    # Found data instances indexes for which violations have been found. These are just few found within first 55 iterations

    # When we duplicate the intance(s) at these indexes [13, 234, 486, 495, 504, 881, 957, 1201, 1253, 1303, 1437, 1552, 1619, 1662, 2179, 2246, 2352, 2413, 2547, 2548, 2649, 2700, 2764, 2833, 2922], we got inconsistent output for the follow-up input
    #print(rfm_Source.loc[[13, 234, 486, 495, 504, 881, 957, 1201, 1253, 1303, 1437, 1552, 1619, 1662, 2179, 2246, 2352, 2413, 2547, 2548, 2649, 2700, 2764, 2833, 2922]])
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR4", rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print(rfm_Source.loc[[350,  767,  847]])
    #print(rfm_Followup.loc[[350, 767, 847]])

    # =========MR#5: Adding datapoint(s) with 0 features' values should not change the output=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR5",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source,rfm_Followup,left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)

    # =========MR#6: Consistent with reprediction=============#
    rfm_SourceMR6, rfm_df_scaled_SourceMR6, dbScanModel_SourceMR6 = dbScanAlgorithm("MR6", rfm_Source)
    rfm_Followup, rfm_df_scaled_Followup, dbScanModel_Followup = dbScanAlgorithm("MR6Followup", rfm_Source)
    mergedDataFrame = pd.merge(rfm_SourceMR6,rfm_Followup,left_index=True, right_index=True)
    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    print("inconsistentOutDataRows = ", inconsistentOutDataRows)

    # =========MR#7: Shifting the data set features by a constant i.e., x + 2=============#
    ##To find the exact/actual data rows (their indexes) for which the model produces inconsistent results for the source and follow-up executions
    ## I run it for a lot of iterations with different values but found no inconsistency
    #valueToIncrement = 0.00099
    #conflictDataInstances = []
    #for rowIndex in range(1001):
    #  print("Iteration#1: ", rowIndex)
    #  rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR7", rfm_Source,rowIndex,valueToIncrement)
    #  mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #  inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #  print("Inconsistent = ", (inconsistentOutDataRows.head()).to_string())
    #  if len(inconsistentOutDataRows)>0:
    #       conflictDataInstances.append(valueToIncrement)
    #       print("Value for which we got inconsistent outputs = ", valueToIncrement)
    #  valueToIncrement = valueToIncrement + 0.00099
    #print("Values for which we got different outputs = ", conflictDataInstances)
    # ========================================================#

    # =========MR#8: Scaling the data set features by a constant i.e., x * 2=============#
    ##To find the exact/actual data rows (thier indexes) for which the model produces inconsistent results for the source and follow-up executions
    ## ## I run it for a lot of iterations, for lot of values we got inconsistent results e.g.,0.91,7.31,7.32,7.33,7.34,7.35,7.36 etc.
    #valueToIncrement = 0.0099
    #conflictDataInstances = []
    #for rowIndex in range(1001):
    #    print("Iteration#1: ", rowIndex)
    #    rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR8", rfm_Source,rowIndex,valueToIncrement)
    #    mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #    inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #    print("Inconsistent = ", (inconsistentOutDataRows.head()).to_string())
    #    if len(inconsistentOutDataRows)>0:
    #        conflictDataInstances.append(valueToIncrement)
    #        print("Value for which we got inconsistent outputs = ", valueToIncrement)
    #    valueToIncrement = valueToIncrement + 0.0099
    #print("Values for which we got different outputs = ", conflictDataInstances)
    # ========================================================#
    # ==========Actual MR Execution=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR8", rfm_Source,-1,7.31)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head()).to_string())
    #print(rfm_Source.loc[[0,1,2]])
    #print(rfm_Followup.loc[[0,1,2]])

    # =========MR#9:MR_replace=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR9",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head(20)).to_string())

    # =========MR#10:Swapping the features=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR10")
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)

    # =========MR#11: Adding uninformative attribute(s)=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR11",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)

    # =========MR#12:Reversing the data-points=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR12",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)

    # =========MR#13: Multiple all the features with -1=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR13",rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)

    # =========MR#14: Enhancing the compactness of specific cluster=============#
    #rfm_Followup, rfm_df_scaled_Followup, dbScanModel = dbScanAlgorithm("MR14", rfm_Source)
    #mergedDataFrame = pd.merge(rfm_Source, rfm_Followup, left_index=True, right_index=True)
    #inconsistentOutDataRows = mergedDataFrame[mergedDataFrame["Cluster_Id_x"] != mergedDataFrame["Cluster_Id_y"]]
    #print("inconsistentOutDataRows = ", inconsistentOutDataRows)
    #print("inconsistentOutDataRows indexes = ", inconsistentOutDataRows.index)
    #print("inconsistentOutDataRows at specific index = ", (inconsistentOutDataRows.head(20)).to_string())
    #print(rfm_Source.loc[[1532]])
    #print(rfm_Followup.loc[[1532]])