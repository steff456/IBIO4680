import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as sm

print("--------- Loading Images ---------")
#The images are imported to the workspace
route = "./Files/"
rog_or = cv2.imread(route + "rog.png")
edg_or = cv2.imread(route + "Edgar.png")

print("--------- Hybrid Image ---------")
#Low pass filter and high pass filters of the images
rog_lp = cv2.GaussianBlur(rog_or, (51,51), 8)
rog_hf = cv2.subtract(rog_or, rog_lp)

edg_lp = cv2.GaussianBlur(edg_or, (25,25), 50)
edg_hf = cv2.subtract(edg_or, edg_lp)

#The hybrid image is created.
H = rog_hf + edg_lp

#Obtain each channel
b,g,r = cv2.split(H)

#Create RGB image from scratch
H_rgb = cv2.merge([r,g,b])    

#Show the hybrid image
plt.imshow(H_rgb)
plt.show()

print("--------- Saving the Hybrid Image to the Files Directory ---------")
#Saves the image in the Files directory.
cv2.imwrite(route + 'hybrid.png', H)

print("--------- Blended Image ---------")
#The list for the gaussian pyramids for both images is initialized
gp_rog = [rog_or]
gp_edg = [edg_or]

#Create the gaussian pyramid for each one of the images
for i in range(0,4):
    #First image
    col, row, _ = gp_rog[-1].shape
    rog_act = cv2.pyrDown(gp_rog[-1]) #, dstsize = (2//col, 2//row))
    gp_rog.append(rog_act)
    #Second image
    col, row, _ = gp_edg[-1].shape
    edg_act = cv2.pyrDown(gp_edg[-1]) #, dstsize = (2//col, 2//row))
    gp_edg.append(edg_act)

#The list fot the lapacian pyramids for both images is initialized
lp_rog = []
lp_edg = []

#Create the laplacian pyramid for each one of the images
for i in range(0,4):
    #First image
    x_rog = gp_rog[i]
    px_rog = gp_rog[(i+1)]
    fgx_rog = cv2.pyrUp(px_rog)
    if x_rog.shape != fgx_rog.shape:
        fgx_rog = fgx_rog[:,:-1,:]
    rog_act = cv2.subtract(x_rog, fgx_rog)
    lp_rog.append(rog_act)
    print(x_rog.shape) 
    print(fgx_rog.shape)
    #Second image
    x_edg = gp_edg[i]
    px_edg = gp_edg[(i+1)]
    fgx_edg = cv2.pyrUp(px_edg)
    if x_edg.shape != fgx_edg.shape:
        fgx_edg = fgx_edg[:,:-1,:]
    edg_act = cv2.subtract(x_edg, fgx_edg)
    lp_edg.append(edg_act)
    print(x_edg.shape) 
    print(fgx_edg.shape)

#Create the image using half and half
