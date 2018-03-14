from segmentByClustering import segmentByClustering
import os
import os.path as osp
import numpy as np
import scipy.io as sio
from PIL import Image
count = 1
ks = range(3,31,3)
for root, dirs, files in os.walk('.', topdown=False):
    print(root)
    if(root.startswith('./BSR/BSDS500/data/images/train')):
        for f in files:
            if f.endswith('.jpg'):
                s = root + '/' + f
                im = Image.open(s)
                im.load()
                img_act = np.asarray(im, dtype='uint8')
                print('********')
                print(f)
                print(count)
                nsf = np.zeros((len(ks,)), dtype = np.object)
                nss = np.zeros((len(ks),), dtype = np.object)
                i = 0
                for pos in ks:
                    print((i+1))
                    print('----------')
                    nsf[i] = segmentByClustering(img_act, 'lab+xy', 'gmm', pos)
                    nss[i] = segmentByClustering(img_act, 'rgb+xy', 'gmm', pos)
                    i+=1
                name = f.split('.')
                d = 'results_lab'
                dd = 'results_rgb'
                if not osp.exists(d):
                    os.mkdir(d)
                if not osp.exists(dd):
                    os.mkdir(dd)
                n1 = d + '/' + name[0] + '.mat'
                n2 = dd + '/' + name[0] + '.mat'
                sio.savemat(n1, {'segs':nsf})
                sio.savemat(n2, {'segs':nss})
                count +=1


