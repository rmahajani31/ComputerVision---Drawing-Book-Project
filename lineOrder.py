import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize,dilation
from skimage.util import invert
import skimage.transform as T
import skimage.draw as D
from tqdm import trange

def cline(x0,y0,angle,dist,img):
    if y0<0:
        y0 = 0
        x0 = int((y0*np.sin(angle)-dist)/-np.cos(angle))
    if y0>=img.shape[0]:
        y0 = img.shape[0]-1
        x0 = int((y0*np.sin(angle)-dist)/-np.cos(angle))
    return x0,y0

def eline(lr,lc,img,s=3):
    sd = s+s+1
    selem = np.ones((sd,sd),dtype=np.uint8)
    cvs = np.zeros(img.shape,dtype=np.uint8)
    cvs[lr,lc] = 1
    cvs = dilation(cvs,selem)
    return np.nonzero(cvs)

def eline2(lr,lc,img,s=5):
    cvs = np.zeros(img.shape,dtype=np.uint8)
    for i in range(len(lr)):
        x = lr[i]
        y = lc[i]
        cvs[x:x+s,y:y+s] = 1
    return np.nonzero(cvs)

def hough(img):
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 720)
    lines,theta,d = T.hough_line(img,theta=tested_angles)
    origin = np.array((0, img.shape[1]))
    j = 0
    _,lang,ldist = T.hough_line_peaks(lines, theta, d)
    bl = []
    lr = []
    lc = []
    for i in range(lang.shape[0]):
        angle = lang[i]
        dist = ldist[i]
        ori2 = np.array((0,img.shape[1]-1))
        x0,x1 = ori2
        y0, y1 = ((dist - ori2 * np.cos(angle)) / np.sin(angle)).astype(np.int32)
        x0,y0 = cline(x0,y0,angle,dist,img)
        x1,y1 = cline(x1,y1,angle,dist,img)
        lri,lci = D.line(y0,x0,y1,x1)
        lr.append(lri)
        lc.append(lci)
    return lr,lc

def lineOrder(img,n_steps,line_thresh=50,len_thresh=60,wsz=10):
    img = invert(img)
    img[img<=127] = 0
    img[img>127] = 1
    i2 = img.copy()*255
    i2 = cv2.cvtColor(i2,cv2.COLOR_GRAY2RGB)
    lr,lc = hough(img)
    skel = skeletonize(img).astype(np.float32)
    skt = skel.copy()
    S = min(line_thresh,len(lr))
    rlist = []
    clist = []
    for sel in trange(S):
        lrs = []
        lcs = []
        resp = []
        for i in range(len(lr)):
            lsz = len(lr[i])
            r = []
            for j in range(lsz):
                x = lr[i][j]
                y = lc[i][j]
                r.append(1 if np.sum(skt[x:x+wsz,y:y+wsz])>0 else 0)
            r = np.array(r)
            dp = np.zeros(r.shape,np.uint8)
            dp[0] = r[0]
            for j in range(1,r.shape[0]):
                dp[j] = r[j]*(dp[j-1]+1)
            mxi = np.argmax(dp)
            lrs.append(lr[i][mxi-dp[mxi]+1:mxi+1])
            lcs.append(lc[i][mxi-dp[mxi]+1:mxi+1])
            resp.append(dp[mxi])
        bind = np.argsort(resp)[-1]
        if resp[bind]<len_thresh:
            break
        lre,lce = eline2(lrs[bind],lcs[bind],img,s=wsz)
        lrp,lcp = eline2(lrs[bind],lcs[bind],img,s=10)
        skt[lre,lce] = 0
        rlist.append(lrp)
        clist.append(lcp)
    splits = list(range(0,len(rlist),len(rlist)//n_steps))
    splits.append(len(rlist))
    rans = []
    cans = []
    for i in range(len(splits)-1):
        rans.append(rlist[splits[i]:splits[i+1]])
        cans.append(clist[splits[i]:splits[i+1]])
    return rans,cans

'''
Example Usage:

img = cv2.imread("bike.jpg",cv2.IMREAD_GRAYSCALE)
i2 = img.copy()*255
i2 = cv2.cvtColor(i2,cv2.COLOR_GRAY2RGB)
a,b=lineOrder(img,3)
for i in range(len(a)):
    for j in range(len(a[i])):
        i2[a[i][j],b[i][j]] = [255,0,0]
plt.imshow(i2)
plt.show()
'''
