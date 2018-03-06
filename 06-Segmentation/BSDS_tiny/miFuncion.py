#!/usr/local/bin/python3

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

#function to load images from folder as np.array
def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="uint8" )
    return data

#main function performing segmentation
def segmentByClustering(rgbImage, featureSpace, clusteringMethod, numberOfClusters):
    #verify the image is in np array, if not, it is casted
    rgbImage=np.array(rgbImage)
    #color and space maximum normalization values. It will change the representativeness of each channel
    colmax=5
    spacemax=50
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
        print('Estoy creando el clusterizador')
        hierClus=cluster.AgglomerativeClustering(n_clusters=numberOfClusters,affinity='euclidean')
        print('Estoy prediciendo los datos')
        assig=hierClus.fit_predict(imager)
        print('Ya acabe de predecir')
        seg=np.reshape(assig,(image.shape[0],image.shape[1]))
        return seg
    elif clusteringMethod=='watershed':
        image=np.mean(image,axis=2)
        local_max = peak_local_max(-1*image, indices=False,num_peaks=numberOfClusters,num_peaks_per_label=1)
        markers=ndi.label(local_max)[0]
        seg=watershed(image,markers)
        return seg
        
#Evaluation of kmeans in different Spaces

#Creation of a dictionary containing the images
image_list={}
for filename in glob.glob('./*.jpg'):
    key=filename.split('/')
    filename=key[1]
    key=filename.split('.')
    key=key[0]
    image_list[key]=load_image(filename)

#Creation of a dictionary containing the annotations
annotation_list={}
for filename in glob.glob('./*.mat'):
    key=filename.split('/')
    filename=key[1]
    key=filename.split('.')
    key=key[0]
    gt=sio.loadmat(filename)
    annotation_list[key]=[]
    for i in range(3):
        segm=gt['groundTruth'][0,i][0][0]['Segmentation']
        annotation_list[key].append(segm)



#evaluation--------------------------------------------------------------------------------------
#for each clustering method all space colors will be used for each image
#Responses for k means method
resp=np.zeros([len(image_list),6,4])

#to fill the Kmeans table = resp[:,:,0]----------------------------------------------------------
contador=0
for keyim in image_list.keys():
    image=image_list[keyim]
    anot1=annotation_list[keyim][0]
    anot2=annotation_list[keyim][1]
    anot3=annotation_list[keyim][2]
    #Different segmentation depending on feature space
    a=segmentByClustering(image,'rgb','kmeans',5)
    b=segmentByClustering(image,'lab','kmeans',5)
    c=segmentByClustering(image,'hsv','kmeans',5)
    d=segmentByClustering(image,'rgb+xy','kmeans',5)
    e=segmentByClustering(image,'lab+xy','kmeans',5)
    f=segmentByClustering(image,'hsv+xy','kmeans',5)
    #number of objects in annotations for subject1,2 and 3
    numanot1=np.unique(anot1).shape[0]
    numanot2=np.unique(anot2).shape[0]
    numanot3=np.unique(anot3).shape[0]
    #for segmentation "a",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "a", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "a", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))    
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,0,0]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "b",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,1,0]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "c",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,2,0]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "d",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,3,0]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "e",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,4,0]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "f",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,5,0]= np.mean([jac1a,jac2a,jac3a])
    contador=contador+1
    print(contador)

#to fill the GMM table = resp[:,:,1]----------------------------------------------------------
contador=0
for keyim in image_list.keys():
    image=image_list[keyim]
    anot1=annotation_list[keyim][0]
    anot2=annotation_list[keyim][1]
    anot3=annotation_list[keyim][2]
    #Different segmentation depending on feature space
    a=segmentByClustering(image,'rgb','gmm',5)
    b=segmentByClustering(image,'lab','gmm',5)
    c=segmentByClustering(image,'hsv','gmm',5)
    d=segmentByClustering(image,'rgb+xy','gmm',5)
    e=segmentByClustering(image,'lab+xy','gmm',5)
    f=segmentByClustering(image,'hsv+xy','gmm',5)
    #number of objects in annotations for subject1,2 and 3
    numanot1=np.unique(anot1).shape[0]
    numanot2=np.unique(anot2).shape[0]
    numanot3=np.unique(anot3).shape[0]
    #for segmentation "a",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "a", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "a", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))    
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,0,1]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "b",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,1,1]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "c",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,2,1]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "d",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,3,1]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "e",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,4,1]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "f",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,5,1]= np.mean([jac1a,jac2a,jac3a])
    contador=contador+1
    print(contador)

#to fill the watershed table = resp[:,:,3]----------------------------------------------------------
contador=0
for keyim in image_list.keys():
    image=image_list[keyim]
    anot1=annotation_list[keyim][0]
    anot2=annotation_list[keyim][1]
    anot3=annotation_list[keyim][2]
    #Different segmentation depending on feature space
    a=segmentByClustering(image,'rgb','watershed',5)
    b=segmentByClustering(image,'lab','watershed',5)
    c=segmentByClustering(image,'hsv','watershed',5)
    d=segmentByClustering(image,'rgb+xy','watershed',5)
    e=segmentByClustering(image,'lab+xy','watershed',5)
    f=segmentByClustering(image,'hsv+xy','watershed',5)
    #number of objects in annotations for subject1,2 and 3
    numanot1=np.unique(anot1).shape[0]
    numanot2=np.unique(anot2).shape[0]
    numanot3=np.unique(anot3).shape[0]
    #for segmentation "a",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    print(Jaccardsa)
    #for segmentation "a", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    print(Jaccardsa)
    #for segmentation "a", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))    
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    print(Jaccardsa)
    resp[contador,0,3]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "b",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,1,3]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "c",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,2,3]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "d",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,3,3]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "e",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,4,3]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "f",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,5,3]= np.mean([jac1a,jac2a,jac3a])
    contador=contador+1
    print(contador)


#to fill the hierarchical table = resp[:,:,2]----------------------------------------------------------
contador=0
for keyim in image_list.keys():
    image=image_list[keyim]
    anot1=annotation_list[keyim][0]
    anot2=annotation_list[keyim][1]
    anot3=annotation_list[keyim][2]
    #Different segmentation depending on feature space
    a=segmentByClustering(image,'rgb','hierarchical',5)
    b=segmentByClustering(image,'lab','hierarchical',5)
    c=segmentByClustering(image,'hsv','hierarchical',5)
    d=segmentByClustering(image,'rgb+xy','hierarchical',5)
    e=segmentByClustering(image,'lab+xy','hierarchical',5)
    f=segmentByClustering(image,'hsv+xy','hierarchical',5)
    #number of objects in annotations for subject1,2 and 3
    numanot1=np.unique(anot1).shape[0]
    numanot2=np.unique(anot2).shape[0]
    numanot3=np.unique(anot3).shape[0]
    #for segmentation "a",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    print(Jaccardsa)
    #for segmentation "a", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    print(Jaccardsa)
    #for segmentation "a", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(a).shape[0]):
            segmentationi=(a==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))    
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    print(Jaccardsa)
    resp[contador,0,2]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "b",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "b", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(b).shape[0]):
            segmentationi=(b==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,1,2]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "c",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "c", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(c).shape[0]):
            segmentationi=(c==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,2,2]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "d",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "d", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(d).shape[0]):
            segmentationi=(d==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,3,2]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "e",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "e", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(e).shape[0]):
            segmentationi=(e==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,4,2]= np.mean([jac1a,jac2a,jac3a])
    #for segmentation "f",human1
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac1a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human2
    Jaccardsa=[]
    for i in range(numanot1+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac2a=np.mean(np.sort(Jaccardsa)[-5:])
    #for segmentation "f", human3
    Jaccardsa=[]
    for i in range(numanot3+1)[1:]:
        anotacioni=(anot1==i)
        Janoti=[]
        for j in range(np.unique(f).shape[0]):
            segmentationi=(f==j)
            intersec=sum(sum(np.logical_and(anotacioni,segmentationi).astype(int)))
            union=sum(sum(np.logical_or(anotacioni,segmentationi).astype(int)))
            if union !=0:
                Janoti.append(intersec/union)
        Jaccardsa.append(np.max(Janoti))
    jac3a=np.mean(np.sort(Jaccardsa)[-5:])
    resp[contador,5,2]= np.mean([jac1a,jac2a,jac3a])
    contador=contador+1
    print(contador)

















a=load_image('55075.jpg')
segmentByClustering(a,'lab+xy','kmeans',5)







