import sys
from skimage import color
from skimage import io
import os
import os.path as osp
import numpy as np
sys.path.append('lib/python')
from fbCreate import fbCreate
from fbRun import fbRun
from computeTextons import computeTextons
from assignTextons import assignTextons

#Function that creates the texton representation to the images that are contained
#in the data directory. 
#numImages are the number of train images
#N is the number of categories 0 < N < 26
#n is the number of columns
#m is the number of rows
#k is the number of clusters
def create_texton_representation(numImages = 10, N = 25, n = 100, m = 100, k = 50):
    #Creates the filter bank
    fb = fbCreate()

    #Load 'numImages' train images of the N categories 0<N<26
    train = {}
    test = {}
    print('---------- Importing dataset to workspace ----------')
    for root, dirs, files in os.walk('.', topdown=False):
        for f in files:
            if f.endswith('.jpg'):
                info = root.split('_')
                imAct=color.rgb2gray(io.imread(root + '/' + f))
                nameAct = info[1]
                train_k = list(train.keys())
                test_k = list(test.keys())
                if nameAct not in train_k:
                    train[nameAct] = [imAct[0:n:1,0:m:1]]
                elif nameAct in train_k and len(train[nameAct]) < numImages:
                    l = train[nameAct]
                    l.append(imAct[0:n:1,0:m:1])
                elif nameAct not in test_k:
                    test[nameAct] = [imAct[0:n:1,0:m:1]]
                else:
                    l = test[nameAct]
                    l.append(imAct[0:n:1,0:m:1])
            print(f)
            print(root)
            print('----------')

    print('---------- Applying Filterbank ----------')
    #Apply filterbank to train images
    res = np.zeros(imAct[0:n:1,0:m:1].shape)
    for key in train.keys():
        act = train[key]
        for im in act:
            res = np.hstack((res,im))
        print(key)

    print('---------- Calculating Filterbank Responses ----------')
    filterResponses = fbRun(fb,res[1:])

    print('---------- Computing Textons with ' + k + 'Centroids  ----------')
    #Computer textons from filter
    map, textons = computeTextons(filterResponses, k)

    print('---------- Saving Calculated Textons ----------')
    #Save the textons
    name = "map_textons.npy"
    if osp.exists(name):
        os.system('rm -rf ' + name)

    _ = [map, textons]
    np.save(name, _)

    print('---------- Calculate the Texton Representation to the Complete Dataset ----------')
    #Calculate texton representation with current texton dictionary
    train_texton = {}
    test_texton = {}
    for key in train.keys():
        act_train = train[key]
        act_test = test[key]
        for im in act_train:
            if key not in train_texton:
                train_texton[key] = [assignTextons(fbRun(fb,im),textons.transpose())]
            else:
                l = train_texton[key]
                l.append([assignTextons(fbRun(fb,im),textons.transpose())])
        for im in act_test:
            if key not in test_texton:
                test_texton[key] = [assignTextons(fbRun(fb,im),textons.transpose())]
            else:
                l = test_texton[key]
                l.append([assignTextons(fbRun(fb,im),textons.transpose())])
        print(key)

    #Save texton representation
    print('---------- Saving the Texton Representations in a File on Workspace ----------')
    name = "texton_representation.npy"
    if osp.exists(name):
        os.system('rm -rf ' + name)

    _ = [train_texton, test_texton]
    np.save(name, _)

    return train_texton, test_texton