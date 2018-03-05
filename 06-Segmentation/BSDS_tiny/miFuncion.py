#!/usr/local/bin/python3
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
# gt=sio.loadmat('55075.mat')

# #Load segmentation from third human
# segm=gt['groundTruth'][0,2][0][0]['Segmentation']
# plt.imshow(segm, cmap=plt.get_cmap('summer'))
# plt.show()




def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def segmentByClustering(rgbImage, featureSpace, clusteringMethod, numberOfClusters):

    rgbImage=np.array(rgbImage)
    
    if featureSpace == 'rgb':
        image=rgbImage
    elif featureSpace=='lab':
        image= color.rgb2lab(rgbImage)
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    elif featureSpace=='hsv':
        image= color.rgb2hsv(rgbImage)
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    elif featureSpace=='rgb+xy':
        image=rgbImage
        xcoord=np.matlib.repmat(np.array(range(image.shape[1])),image.shape[0],1)
        ycoord=np.matlib.repmat(np.transpose([np.array(range(image.shape[0]))]),1,image.shape[1])
        image=np.stack((image[:,:,0],image[:,:,1],image[:,:,2],xcoord,ycoord),axis=2)
    elif featureSpace=='lab+xy':
        image= color.rgb2lab(rgbImage)
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        xcoord=np.matlib.repmat(np.array(range(image.shape[1])),image.shape[0],1)
        ycoord=np.matlib.repmat(np.transpose([np.array(range(image.shape[0]))]),1,image.shape[1])
        image=np.stack((image[:,:,0],image[:,:,1],image[:,:,2],xcoord,ycoord),axis=2)
    elif featureSpace=='hsv+xy':
        image= color.rgb2hsv(rgbImage)
        image=cv2.normalize(image,np.zeros((image.shape),dtype=np.uint8),alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
        xcoord=np.matlib.repmat(np.array(range(image.shape[1])),image.shape[0],1)
        ycoord=np.matlib.repmat(np.transpose([np.array(range(image.shape[0]))]),1,image.shape[1])
        image=np.stack((image[:,:,0],image[:,:,1],image[:,:,2],xcoord,ycoord),axis=2)

    if clusteringMethod=='kmeans':
    	imager=np.reshape(image,(1,image.shape[0]*image.shape[1],image.shape[2]))[0]
    	kmeans=KMeans(n_clusters=numberOfClusters,random_state=0).fit(imager)
    	assig=kmeans.labels_
    	seg=np.reshape(assig,(image.shape[0],image.shape[1]))
    	plt.imshow(seg,cmap=plt.get_cmap('tab20b'))
    	plt.show()
    	return seg
    elif clusteringMethod=='gmm':
        imager=np.reshape(image,(1,image.shape[0]*image.shape[1],image.shape[2]))[0]
        gmm=mixture.GaussianMixture(n_components=numberOfClusters,covariance_type='full').fit(imager)
        assig=gmm.predict(imager)
        seg=np.reshape(assig,(image.shape[0],image.shape[1]))
        plt.imshow(seg,cmap=plt.get_cmap('tab20b'))
        plt.show()
        return seg
    elif clusteringMethod=='hierarchical':
        imager=np.reshape(image,(1,image.shape[0]*image.shape[1],image.shape[2]))[0]
        print('Estoy creando el clusterizador')
        hierClus=cluster.AgglomerativeClustering(n_clusters=numberOfClusters,affinity='euclidean')
        print('Estoy prediciendo los datos')
        assig=hierClus.fit_predict(imager)
        print('Ya acabe de predecir')
        seg=np.reshape(assig,(image.shape[0],image.shape[1]))
        plt.imshow(seg,cmap=plt.get_cmap('tab20b'))
        plt.show()
    elif clusteringMethod=='watershed':
    	print('AIuda con este metodo plox')





a=load_image('55075.jpg')
segmentByClustering(a,'lab','kmeans',10)






