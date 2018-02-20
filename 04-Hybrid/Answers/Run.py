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
#Low pass filter => High pass image
rog_lp = cv2.GaussianBlur(rog_or, (51,51), 8)
rog_hf = cv2.subtract(rog_or, rog_lp)

edg_lp = cv2.GaussianBlur(edg_or, (25,25), 50)
edg_hf = cv2.subtract(edg_or, edg_lp)
#cv2.imshow('Gaussian Blurring', r_lp)

H = rog_hf + edg_lp
#High pass filter 

cv2.imwrite(route + 'hybrid.png', H)

cv2.startWindowThread()
cv2.namedWindow('A')
cv2.imshow('A', H)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("--------- Blended Image ---------")
