import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os.path as osp
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('lib/python')
from confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import time
#route_map_tex = 'map_textons.npy'
#route_texton_rep = 'texton_representation.npy'

#Function that classify saved data or new dictionaries of train and test
#k is the number of centroids
#route_texton_rep is the saved file
#train_texton is new dict with texton representation for train data
#test_texton is new dict with texton representation for the test data
def classify_RF(k=50, route_texton_rep = '', train_texton = {}, test_texton = {}):
    #map, textons = np.load(route_map_tex, encoding='latin1')
    #Time start
    start_t = time.time()
    #Loads data from the route
    if osp.exists(route_texton_rep):
        print('---------- Importing Data ----------')
        train_texton, test_texton = np.load(route_texton_rep, encoding = 'latin1')

    #verifies there's data to classify
    if len(train_texton) == 0 or len(test_texton) == 0:
        print('STOP - Error in the data!')
        return 0

    #Defines the function to create histograms
    def histc(X, bins):
        map_to_bins = np.digitize(X,bins)
        r = np.zeros(bins.shape)
        for i in map_to_bins:
            r[i-1] += 1
        return np.array(r)

    #Data = histogram from train data
    train_data = []
    #Labels of the train data
    train_labels = []

    print('---------- Calculate Histograms for the Train Data ----------')
    #Calculates the histograms for the train data
    for key in train_texton.keys():
        l = train_texton[key]
        for i in range(0,len(l)):
            if i == 0:
                m_act = l[i]
            else:
                m_act = l[i][0]
            act_hist = histc(m_act.flatten(), np.arange(k))/m_act.size
            train_data.append(act_hist)
            train_labels.append(key)

    #Data = histogram from test data
    test_data = []
    #Labels of the test data
    test_labels = []

    print('---------- Calculate Histograms for the Test Data ----------')
    #Calculates the histograms for the test data
    for key in test_texton.keys():
        l = test_texton[key]
        for i in range(0,len(l)):
            if i == 0:
                m_act = l[i]
            else:
                m_act = l[i][0]
            act_hist = histc(m_act.flatten(), np.arange(k))/m_act.size
            test_data.append(act_hist)
            test_labels.append(key)

    print('---------- Create the KN Classifier ----------')
    #Create KN classifier
    rf = RandomForestClassifier(max_depth = 2, random_state = 0)

    print('---------- Fit Train Data to Classifier ----------')
    #Fit with training data
    rf.fit(train_data, train_labels)

    print('---------- Predict Test Data ----------')
    #Predict the test data
    res = rf.predict(test_data)

    print('---------- Calculating Number of True Positives ----------')
    cnf = confusion_matrix(test_labels,res)
    plot_confusion_matrix(cnf, classes = list(train_texton.keys()), normalize = True)
    total_t = (time.time()-start_t)
    print('---------- Total time = '+ str(total_t) + ' ----------')
    plt.show()

    return cnf