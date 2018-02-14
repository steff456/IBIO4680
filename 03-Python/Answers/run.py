#!/usr/bin/env python
import os
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import cv2
import scipy.io as sio
import time

#Download of the database
if not osp.exists('BSR'):
    print('------ Downloading database ------')
    os.system('wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz')
    print('------ Unzip database ------')
    os.system('tar -czvf BSR_bsds500.tgz BSR')

start_t = time.time()

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

#Generates a random number between 1 and 15
N = random.randint(7,15)

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
print('------ Resize + New Folder ------')
while(n>0):
    #Actual image
    num = random.randint(1,count)
    #Get the images and groundtruths
    mat_act = sio.loadmat(seen[seen_keys[num]][0])['groundTruth']
    img_act = cv2.imread(seen[seen_keys[num]][1]) 
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

#Number of images for plotting
P = N

print('------ Plotting '+ str(P) +' image(s) ------')

# Plot of the images
fig = plt.figure()

j = 1
for i in range(1,P+1):
    #Original image
    num = random.randint(0,(len(final_d)-1))
    keys = list(final_d.keys())
    key = keys[num]
    sub_fig = fig.add_subplot(P,3,j)
    img_act = final_d[key][0]
    img_p = plt.imshow(img_act)
    j+=1
    #gt
    sub_fig = fig.add_subplot(P,3,j)
    img_act = final_d[key][1]
    img_p = plt.imshow(img_act)    
    j+=1
    #gt
    sub_fig = fig.add_subplot(P,3,j)
    img_act = final_d[key][2]
    img_p = plt.imshow(img_act)
    j+=1
    final_d.pop(key, None)

total_t = (time.time()-start_t)
plt.show()

print('------ '+ str(total_t) + ' seconds ------')
