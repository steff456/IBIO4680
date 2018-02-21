import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as sm

#Definition for graphing cv2 in matplotlib
def graph_cv2(im):
    #Decompose the channels
    b,g,r = cv2.split(im)
    #Create RGB image from scratch
    im_rgb = cv2.merge([r,g,b])
    return im_rgb  

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

#Pass to RGB
H_rgb = graph_cv2(H)    

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

print("--------- Creating Gaussian Pyramid ---------")
#Create the gaussian pyramid for each one of the images
for i in range(0,5):
    #First image
    col, row, _ = gp_rog[-1].shape
    rog_act = cv2.pyrDown(gp_rog[-1])
    gp_rog.append(rog_act)
    #Second image
    col, row, _ = gp_edg[-1].shape
    edg_act = cv2.pyrDown(gp_edg[-1])
    gp_edg.append(edg_act)

print("--------- Creating Laplacian Pyramid ---------")
#The list fot the lapacian pyramids for both images is initialized
lp_rog = []
lp_edg = []

#Create the laplacian pyramid for each one of the images - 4 levels
for i in range(0,5):
    #First image
    x_rog = gp_rog[i]
    px_rog = gp_rog[(i+1)]
    fgx_rog = cv2.pyrUp(px_rog)
    if x_rog.shape != fgx_rog.shape: #In the case of odd sizes
        fgx_rog = fgx_rog[:,:-1,:]
    rog_act = cv2.subtract(x_rog, fgx_rog)
    lp_rog.append(rog_act)
    #Second image
    x_edg = gp_edg[i]
    px_edg = gp_edg[(i+1)]
    fgx_edg = cv2.pyrUp(px_edg)
    if x_edg.shape != fgx_edg.shape: #In the case of odd sizes
        fgx_edg = fgx_edg[:,:-1,:]
    edg_act = cv2.subtract(x_edg, fgx_edg)
    lp_edg.append(edg_act)

print("--------- Joining Half Images per Level ---------")
#Join each half on each level of the pyramid and save them in new list L
L = []
for i in range(0,5):
    edg = lp_edg[i]
    rog = lp_rog[i]
    col, row, _ = edg.shape
    l_act = np.hstack((edg[:,0:int(row/2),:],rog[:,int(row/2):,:]))
    L.append(l_act)

print("--------- Reconstructing Image ---------")
#Reconstruct the blended image
samples = []
B = L[-1]
for i in range(2,6):
    act = L[-i]
    temp = cv2.pyrUp(B)
    if act.shape != temp.shape:
        temp = temp[:,:-1,:]
    B = cv2.add(temp, act)
    samples.append(B)

#Pass to RGB
B_rgb = graph_cv2(B)  

print("--------- Saving the Blended Image to the Files Directory ---------")
#Saves the image in the Files directory.
cv2.imwrite(route + 'blended_pyramid.png', B)

#Show the blended image vs non-blended
col, row, _ = edg_or.shape

nB = np.hstack((edg_or[:,0:int(row/2),:],rog_or[:,int(row/2):,:]))

cv2.imwrite(route + 'non_blended.png', nB)

nB_rgb = graph_cv2(nB)

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(B_rgb)
fig.add_subplot(1,2,2)
plt.imshow(nB_rgb)
plt.show()

print("--------- Showing the gaussian pyramid ---------")
fig = plt.figure()
fig.add_subplot(2,3,1)
fim = gp_rog[-1]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,2)
fim = gp_rog[-2]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,3)
fim = gp_rog[-3]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,4)
fim = gp_rog[-4]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,5)
fim = gp_rog[-5]
plt.imshow(graph_cv2(fim))
plt.show()

print("--------- Showing the blended pyramid ---------")
fig = plt.figure()
fig.add_subplot(2,3,1)
fim = samples[-1]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,2)
fim = samples[-2]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,3)
fim = samples[-3]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,4)
fim = samples[-4]
plt.imshow(graph_cv2(fim))
fig.add_subplot(2,3,5)
fim = samples[-5]
plt.imshow(graph_cv2(fim))
plt.show()

