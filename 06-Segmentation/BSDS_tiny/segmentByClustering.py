#!/usr/local/bin/python3

#main function performing segmentation
def segmentByClustering(rgbImage, featureSpace, clusteringMethod, numberOfClusters):
    #importing necessary packages and libraries
    import numpy as np
    from skimage import io, color
    import cv2
    import matplotlib.pyplot as plt
    import numpy.matlib
    from sklearn.cluster import KMeans
    import scipy.io as sio
    from PIL import Image
    from sklearn import mixture
    from sklearn import cluster
    import time
    from scipy import ndimage as ndi
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    import glob
    import math
    
    #verify the image is in np array, if not, it is casted
    rgbImage=np.array(rgbImage)
    #color and space maximum normalization values. It will change the representativeness of each channel
    colmax=5
    spacemax=5
    #Image in RGB
    if featureSpace == 'rgb':
        image=rgbImage
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=colmax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    #convert image to lab and resizing the channels to 0-255
    elif featureSpace=='lab':
        image= color.rgb2lab(rgbImage)
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=colmax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    #convert image to hsv and resizing the channels to 0-255
    elif featureSpace=='hsv':
        image= color.rgb2hsv(rgbImage)
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=colmax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)    
    #convert image to rgb+xy and resizing all channels to 0-255
    elif featureSpace=='rgb+xy':        
        image=rgbImage
        #generating the x position and y position matrices to stack to the image
        xcoord=np.matlib.repmat(np.array(range(image.shape[1])),image.shape[0],1)
        ycoord=np.matlib.repmat(np.transpose([np.array(range(image.shape[0]))]),1,image.shape[1])
        #set channels to 0-255 range
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=colmax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        xcoord=cv2.normalize(ycoord,np.zeros((xcoord.shape),dtype=np.uint8),alpha=0, beta=spacemax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        ycoord=cv2.normalize(ycoord,np.zeros((ycoord.shape),dtype=np.uint8),alpha=0, beta=spacemax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        #concatenating image and x,y position matrices
        image=np.stack((image[:,:,0],image[:,:,1],image[:,:,2],xcoord,ycoord),axis=2)    
    #convert image to lab+xy and resizing all channels to 0-255
    elif featureSpace=='lab+xy':
        #convert image to lab colorspace
        image= color.rgb2lab(rgbImage)
        #generating the x position and y position matrices to stack to the image
        xcoord=np.matlib.repmat(np.array(range(image.shape[1])),image.shape[0],1)
        ycoord=np.matlib.repmat(np.transpose([np.array(range(image.shape[0]))]),1,image.shape[1])
        #set channels to 0-255 range
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=colmax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        xcoord=cv2.normalize(xcoord,np.zeros((xcoord.shape),dtype=np.uint8),alpha=0, beta=spacemax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        ycoord=cv2.normalize(ycoord,np.zeros((ycoord.shape),dtype=np.uint8),alpha=0, beta=spacemax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        #concatenating image and x,y position matrices
        image=np.stack((image[:,:,0],image[:,:,1],image[:,:,2],xcoord,ycoord),axis=2)
    elif featureSpace=='hsv+xy':
        #convert image to hsv colorspace
        image= color.rgb2hsv(rgbImage)
        #generating the x position and y position matrices to stack to the image
        xcoord=np.matlib.repmat(np.array(range(image.shape[1])),image.shape[0],1)
        ycoord=np.matlib.repmat(np.transpose([np.array(range(image.shape[0]))]),1,image.shape[1])
        #set channels to 0-255 range
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=colmax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        xcoord=cv2.normalize(xcoord,np.zeros((xcoord.shape),dtype=np.uint8),alpha=0, beta=spacemax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        ycoord=cv2.normalize(ycoord,np.zeros((ycoord.shape),dtype=np.uint8),alpha=0, beta=spacemax,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        #concatenating image and x,y position matrices
        image=np.stack((image[:,:,0],image[:,:,1],image[:,:,2],xcoord,ycoord),axis=2)
    if clusteringMethod=='kmeans':
        imager=np.reshape(image,(1,image.shape[0]*image.shape[1],image.shape[2]))[0]
        kmeans=KMeans(n_clusters=numberOfClusters,random_state=0).fit(imager)
        assig=kmeans.labels_
        seg=np.reshape(assig,(image.shape[0],image.shape[1]))
        return seg
    elif clusteringMethod=='gmm':
        imager=np.reshape(image,(1,image.shape[0]*image.shape[1],image.shape[2]))[0]
        gmm=mixture.GaussianMixture(n_components=numberOfClusters,covariance_type='full').fit(imager)
        assig=gmm.predict(imager)
        seg=np.reshape(assig,(image.shape[0],image.shape[1]))
        return seg
    elif clusteringMethod=='hierarchical':
        imager=np.reshape(image,(1,image.shape[0]*image.shape[1],image.shape[2]))[0]
        hierClus=cluster.AgglomerativeClustering(n_clusters=numberOfClusters,affinity='euclidean')
        assig=hierClus.fit_predict(imager)
        seg=np.reshape(assig,(image.shape[0],image.shape[1]))
        return seg
    elif clusteringMethod=='watershed':
        image=np.mean(image,axis=2)
        local_max = peak_local_max(-1*image, indices=False,num_peaks=numberOfClusters,num_peaks_per_label=1)
        markers=ndi.label(local_max)[0]
        seg=watershed(image,markers)
        return seg
