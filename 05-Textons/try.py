import sys
sys.path.append('code/lib/python')

#Create a filter bank with deafult params
from fbCreate import fbCreate
fb = fbCreate()

#Load sample images from disk
from skimage import color
from skimage import io

imBase1=color.rgb2gray(io.imread('img/person1.bmp'))
imBase2=color.rgb2gray(io.imread('img/goat1.bmp'))

#Set number of clusters
k = 16*8

#Apply filterbank to sample image
from fbRun import fbRun
import numpy as np
filterResponses = fbRun(fb,np.hstack((imBase1,imBase2)))

#Computer textons from filter
from computeTextons import computeTextons
map, textons = computeTextons(filterResponses, k)

#Load more images
imTest1=color.rgb2gray(io.imread('img/person2.bmp'))
imTest2=color.rgb2gray(io.imread('img/goat2.bmp'))

#Calculate texton representation with current texton dictionary
from assignTextons import assignTextons
tmapBase1 = assignTextons(fbRun(fb,imBase1),textons.transpose())
tmapBase2 = assignTextons(fbRun(fb,imBase2),textons.transpose())
tmapTest1 = assignTextons(fbRun(fb,imTest1),textons.transpose())
tmapTest2 = assignTextons(fbRun(fb,imTest2),textons.transpose())

#Check the euclidean distances between the histograms and convince yourself that the images of the goats are closer because they have similar texture pattern

# --> Can you tell why we need to create a histogram before measuring the distance? <---

def histc(X, bins):
    import numpy as np
    map_to_bins = np.digitize(X,bins)
    r = np.zeros(bins.shape)
    for i in map_to_bins:
        r[i-1] += 1
    return np.array(r)

D1 = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
     histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D2 = np.linalg.norm(histc(tmapBase1.flatten(), np.arange(k))/tmapBase1.size - \
     histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)

D3 = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
     histc(tmapTest1.flatten(), np.arange(k))/tmapTest1.size)
D4 = np.linalg.norm(histc(tmapBase2.flatten(), np.arange(k))/tmapBase2.size - \
     histc(tmapTest2.flatten(), np.arange(k))/tmapTest2.size)


#find the max filter size
maxsz = fb[0][0].shape
for filter in fb:
    for scale in filter:
        maxsz = max(maxsz, scale.shape)

maxsz = maxsz[0]

#pad the image 
r = int(math.floor(maxsz/2))
impad = padReflect(im,r)

fim = np.zeros(np.array(fb).shape).tolist()
for i in range(np.array(fb).shape[0]):
    for j in range(np.array(fb).shape[1]):
        if fb[i][j].shape[0]<50:
            fim[i][j] = scipy.signal.convolve2d(impad, fb[i][j], 'same')
        else:
            fim[i][j] = fftconvolve(impad,fb[i][j])
        fim[i][j] = fim[i][j][r:-r,r:-r]
