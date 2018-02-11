import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import cv2
import scipy.io as sio

#Download of the database
if not osp.exists('BSR'):
    print('------ Downloading database ------')
    os.system('wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
    os.system('tar -czvf archive.tar.gz BSR')

#Stores the database images directory
#Dictionary, key: idImage, val: list of image, groundthruths
print('------ Loading information ------')
seen = {}
for root, dirs, files in os.walk('.', topdown=False):
    for f in files:
        if f.endswith('.jpg') or f.endswith('.mat'):
            name = f.split('.')
            if name[0] in seen.keys():
                l= seen[name[0]]
                l.append(root + '/' + f)
            else:
                seen[name[0]] = [root + '/' + f] 

#Finds the number of images of the database 
count = len(seen)

#Generates a random number between 1 and the number of images from the dataset
N = random.randint(1,count)

print('------ Showing ', N, ' images ------')

#Get the keys of the images
seen_keys = list(seen.keys())

#dict for final data
final_d = {}

#Creates the final dir where the images are going to be stored
out = 'Images'
if osp.exists(out):
    os.system('rm -rf ' + out)

os.mkdir(out)

n= N
#Transforms the N images to 256x256 and save them on the new folder
print('------ Resize + Save ------')
while(n>0):
    #Get the images and groundtruths
    mat_act = sio.loadmat(seen[seen_keys[(n-1)]][0])['groundTruth']
    img_act = cv2.imread(seen[seen_keys[(n-1)]][1]) 
    gt = mat_act[0][0][0][0][0]
    gtt = mat_act[0][0][0][0][1]
    #Resize of the images
    img_act = cv2.resize(img_act, (256,256))
    gt = cv2.resize(gt, (256,256))
    gtt = cv2.resize(gtt, (256,256))
    #Converts an array to Image and save them in a new folder
    mpimg.imsave(out + "/" + seen_keys[(n-1)] + ".png", img_act)
    mpimg.imsave(out + "/" + seen_keys[(n-1)] + "_gt.png", gt)
    mpimg.imsave(out + "/" + seen_keys[(n-1)] + "_gtt.png", gtt)
    #Save the images in the final dictionary
    final_d[seen_keys[(n-1)]] = [img_act, gt, gtt]
    n-=1

#Save the dictionary
print('------ Saving information ------')
name = "dict.npy"
if osp.exists(name):
    os.system('rm -rf ' + name)

np.save("dict.npy", final_d)

print('------ Plotting images ------')
#Only plot N<11 images for simplicity
P = N%10 + 1

#Plot of the images
fig = plt.figure()

j = 1
for i in range(1,P+1):
    #Original image
    sub_fig = fig.add_subplot(P,3,j)
    img_act = final_d[seen_keys[(i-1)]][0]
    img_p = plt.imshow(img_act)
    j+=1
    #gt
    sub_fig = fig.add_subplot(P,3,j)
    img_act = final_d[seen_keys[(i-1)]][1]
    img_p = plt.imshow(img_act)    
    j+=1
    #gt
    sub_fig = fig.add_subplot(P,3,j)
    img_act = final_d[seen_keys[(i-1)]][2]
    img_p = plt.imshow(img_act)
    j+=1

plt.show()
