#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:19:45 2020

@author: sujiwosa
"""


# This code is based on http://www.cs.unca.edu/~reiser/imaging/chaincode.html


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from math import sqrt
import io
import numpy as np
from skimage.draw import polygon_perimeter
#https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import pyplot as plt

from itertools import chain
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops, perimeter
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from itertools import islice
import matplotlib.pyplot as plt
from itertools import groupby
from skimage.morphology import convex_hull_image
from skimage import data, img_as_float
from skimage.util import invert
import math
from array import array 
import sklearn.metrics
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import pairwise_distances_argmin
import itertools 
from skimage.filters import threshold_otsu, threshold_local
import skimage
from skimage import img_as_ubyte,measure, feature
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.segmentation import clear_border
from skimage.filters import roberts, sobel,median
#from skimage import io
from skimage.filters import threshold_otsu, threshold_local#,threshold_multiotsu #threshold_adaptive
from scipy.interpolate import UnivariateSpline,NearestNDInterpolator
#import metrics as mt
import matplotlib.pyplot as plt
import matplotlib
import os.path,subprocess
from subprocess import STDOUT,PIPE
import csv
import numpy as np
import sys    
import pandas as pd
import cv2 as cv
from scipy import ndimage, misc
#import matplotlib.pyplot as plt
from pylidc.utils import consensus
import pylidc as pl
import glob
import pydicom as dicom
import copy
#import eval_metric as em
from time import ctime
import csv
import math
import similaritymeasures
from scipy.spatial import ConvexHull  
from sklearn.metrics import f1_score
#import sklearn.metrics.pairwise_distances_argmin
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from scipy.spatial.distance import euclidean

from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score,jaccard_score
from sklearn.metrics import confusion_matrix


plt.ioff()

def get_any_file(path):
    files = glob.glob(path + "/*.dcm")
    if len(files) < 1:
        return None
    return dicom.read_file(files[0])


def getChain(imag):
    for i, row in enumerate(imag):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
#                print(start_point, value)
                break
        else:
            continue
        break
    imag[3:6, 19:22]
    directions = [ 0,  1,  2,
                   7,      3,
                   6,  5,  4]
    dir2idx = dict(zip(directions, range(len(directions))))
    
    change_j =   [-1,  0,  1, # x or columns
                  -1,      1,
                  -1,  0,  1]
    
    change_i =   [-1, -1, -1, # y or rows
                   0,      0,
                   1,  1,  1]
    
    border = []
    chain = []
    curr_point = start_point
#    imagetemp[curr_point[0],curr_point[1]]=128
    
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0]+change_i[idx], start_point[1]+change_j[idx])
        if imag[new_point] != 0: # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
#            imagetemp[curr_point[0],curr_point[1]]=128
    
            break
    
    count = 0
    while curr_point != start_point:
        #figure direction to start search
        b_direction = (direction + 5) % 8 
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0]+change_i[idx], curr_point[1]+change_j[idx])
            if imag[new_point] != 0: # if is ROI
                border.append(new_point)
                chain.append(direction)
                curr_point = new_point
#                imagetemp[curr_point[0],curr_point[1]]=128
                break
        if count == 1000: break
        count += 1
    #print(count)
    #print(chain)
    return border

def drawBorder(border):
    imag=np.zeros((512,512))
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    plt.plot([i[1] for i in border], [i[0] for i in border])      

#def drawConsecBorders(border1,border2):
#    imag=np.zeros((512,512))
#    plt.figure()
#    plt.imshow(imag, cmap='Greys')
#    plt.plot([i[1] for i in border1], [i[0] for i in border1])  
#    plt.plot([i[1] for i in border2], [i[0] for i in border2]) 


def drawConsecBordersB2BOpenCV(border1,border2,borderline,fname,slice1,slice2,patid):
#    filename="/DATA/SG/Suji/implementation/journal7/resultstemp/"
    pathdistance='./outputnew/'+patid+'/a2p/distance/'
    my_dpi=96
#    imag=np.zeros((512,512))
    imag=copy.copy(slice1)
#    plt.axis('off')
#    plt.figure(figsize=(20,10))
#    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi,frameon=False)
    
#    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
#    plt.tight_layout(pad=0, h_pad=None, w_pad=None, rect=None)
#    plt.box(False)
#    plt.axis('off')
    plt.figure()
    
    ax = plt.axes([0,0,1,1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(enable=True, axis='both',tight=True) 


    
#    plt.rcParams['axes.facecolor'] = 'b'
    plt.imshow(imag, cmap='gray')
    x=[]
    y=[]
    for i in border1:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
#    plt.imsave("b2b_1"+fname,plt,cmap='gray')
    x=[]
    y=[]    
    for i in border2:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
#######################################    
#    plt.imsave("b2b_2"+fname,plt,cmap='gray')
    x=[]
    y=[]
    xx=[]
    yy=[]
#    for i,p in border1,borderline:
#    border1=border1[0:100]
#    borderline=borderline[0:100]
    for (a, b) in zip(border1,borderline):        
        y.append(a[0])
        y.append(b[0])
        x.append(a[1])
        x.append(b[1])
        xx.append(x)
        yy.append(y)
        plt.plot(x,y) 
        x=[]
        y=[]
#    plt.savefig("b2b"+fname)
####################################################        
#    plt.box(False)
#    plt.xticks([])
#    plt.yticks([])
#    data = np.fromstring(plt.figure().canvas.tostring_rgb(), dtype=np.uint8, sep='')
#    data = data.reshape(plt.figure().canvas.get_width_height()[::-1] + (3,))
        
#    io_buf = io.BytesIO()
#    plt.savefig(io_buf, format='raw', dpi=my_dpi)
#    io_buf.seek(0)
#    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8), newshape=(800, 800, -1))
#    io_buf.close()    
    
    plt.savefig(pathdistance+"DCB_B2B"+fname, dpi=my_dpi,transparent=True)
#    plt.imsave("b2b"+fname,plt,cmap='gray')
#        print("hello")
#    print()      
def drawConsecBordersB2BOnlyNodulesOpenCV(border1,border2,borderline,slice,E2tnf,fname,patid):
#    border1c=copy.copy(border1) 
##    border2c=copy.copy(border2)
#    borderlinec=copy.copy(borderline)
    pathdistance='./outputnew/'+patid+'/a2p/onlynodulesopencv/'
    my_dpi=96
    imag=np.zeros((512,512,3),dtype=np.uint8)
#    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)
    E2tnfc=copy.copy(E2tnf)
    border1c=[]
    borderlinec=[]
    for i,c in enumerate(E2tnf):
        if(E2tnf[i]==True):
            border1c.append(border1[i])
            borderlinec.append(borderline[i])
            
#        x.append(i[1])
#        y.append(i[0])    
#    border1c[~E2tnfc]=(0,0)
#    borderlinec[~E2tnfc]=(0,0)
##    border2[~E2tnf]=(0,0)    
#    imag=np.zeros((512,512))
#    plt.figure()
#    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in border1c:
#        imag[i[0],i[1]]=[255,0,0]
        imag[i]=[255,255,255]
    for i in borderlinec:
        imag[i]=[255,255,255]
#******************        
    grayImage = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)   
    #(thresh, imag) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    ret, threshed_img = cv2.threshold(grayImage,220, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(pathdistance+"afterthresh1_"+fname, threshed_img) 
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    item=contours[0]
    for i in range(1,len(contours)):
        i=int(i)
        kk=contours[i]
        item=np.concatenate((item, kk))
    contours=item   
    
    hull=cv2.convexHull(np.array(contours,dtype='float32'))
    #cv2.drawContours(img, [hull], -1, (0, 0, 255), 1) 
    contours=[contours]
    for cnt in contours:
        # get convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(imag, [hull], -1, (0, 0, 255), 1)    
#    cv2.imwrite("output.png", img)        
        
#**********
#    grayImage = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)   
#    (thresh, imag) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
#    cv2.imwrite(pathdistance+"afterthresh1_"+fname, imag) 
##    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)    
#    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
#    threshed = cv2.morphologyEx(imag, cv2.MORPH_CLOSE, rect_kernel)
#    cv2.imwrite(pathdistance+"afterthresh2_"+fname, threshed) 
##    cv2.imwrite('thresh2.png', threshed)
#    
#    Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    for cnt in Contours:
#        hull = cv2.convexHull(cnt)
#        cv2.drawContours(imag, [hull], -1, (0, 0, 255), 1) 
#**********************************        
#    cv2.imwrite('output2.png', img)   
     
#        imag[i[0],i[1]]=[0,255,0]        
#        x.append(i[1])
#        y.append(i[0])
#    plt.plot(x,y) 
##    xy = [(a, b) for a in x for b in y]
#    x=[]
#    y=[]
########################################    
#    for i in borderlinec:
#        x.append(i[1])
#        y.append(i[0])
#    plt.plot(x,y)     
##    for i in border2:
##        x.append(i[1])
##        y.append(i[0])
##    plt.plot(x,y) 
#    x=[]
#    y=[]
#    xx=[]
#    yy=[]
#
#  
#   
##    for i,p in border1,borderline:
##    border1=border1[0:100]
##    borderline=borderline[0:100]
###################################################    
#    for (a, b) in zip(border1c,borderlinec):        
#        y.append(a[0])
#        y.append(b[0])
#        x.append(a[1])
#        x.append(b[1])
#        xx.append(x)
#        yy.append(y)
#        plt.plot(x,y) 
#        x=[]
#        y=[]
##################################################        
#    plt.savefig(pathdistance+"onlynodules"+fname, dpi=my_dpi,transparent=True)
#    plt.show()
    #include code for multiple nodules
    cv2.imwrite(pathdistance+"onlynodules_"+fname, imag) 
    return borderlinec,border1c,imag
#        print("hello")
#    print()     
def drawConsecBordersB2BOnlyNodulesBlobOpenCV(border1,border2,borderline,slice,E2tnf,fname,patid):
#    border1c=copy.copy(border1) 
##    border2c=copy.copy(border2)
#    borderlinec=copy.copy(borderline)
    pathdistance='./outputnew/'+patid+'/a2p/onlynodulesopencvblob/'
    my_dpi=96
    imag=np.zeros((512,512,3),dtype=np.uint8)
#    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)
    E2tnfc=copy.copy(E2tnf)
    border1c=[]
    borderlinec=[]
    for i,c in enumerate(E2tnf):
        if(E2tnf[i]==True):
            border1c.append(border1[i])
            borderlinec.append(borderline[i])
            
#        x.append(i[1])
#        y.append(i[0])    
#    border1c[~E2tnfc]=(0,0)
#    borderlinec[~E2tnfc]=(0,0)
##    border2[~E2tnf]=(0,0)    
#    imag=np.zeros((512,512))
#    plt.figure()
#    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in border1c:
#        imag[i[0],i[1]]=[255,0,0]
        imag[i]=[255,255,255]
    for i in borderlinec:
        imag[i]=[255,255,255]
#******************        
    grayImage = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)   
    #(thresh, imag) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    ret, threshed_img = cv2.threshold(grayImage,220, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(pathdistance+"afterthresh1_"+fname, threshed_img) 
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    item=contours[0]
    for i in range(1,len(contours)):
        i=int(i)
        kk=contours[i]
        item=np.concatenate((item, kk))
    contours=item   
    
    hull=cv2.convexHull(np.array(contours,dtype='float32'))
    #cv2.drawContours(img, [hull], -1, (0, 0, 255), 1) 
    contours=[contours]
    for cnt in contours:
        # get convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(imag, [hull], -1, (255, 255, 255), -1)    
        
#    cv2.imwrite("output.png", img)        
        
#**********
#    grayImage = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)   
#    (thresh, imag) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
#    cv2.imwrite(pathdistance+"afterthresh1_"+fname, imag) 
##    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)    
#    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
#    threshed = cv2.morphologyEx(imag, cv2.MORPH_CLOSE, rect_kernel)
#    cv2.imwrite(pathdistance+"afterthresh2_"+fname, threshed) 
##    cv2.imwrite('thresh2.png', threshed)
#    
#    Contours, Hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#    for cnt in Contours:
#        hull = cv2.convexHull(cnt)
#        cv2.drawContours(imag, [hull], -1, (0, 0, 255), 1) 
#**********************************        
#    cv2.imwrite('output2.png', img)   
     
#        imag[i[0],i[1]]=[0,255,0]        
#        x.append(i[1])
#        y.append(i[0])
#    plt.plot(x,y) 
##    xy = [(a, b) for a in x for b in y]
#    x=[]
#    y=[]
########################################    
#    for i in borderlinec:
#        x.append(i[1])
#        y.append(i[0])
#    plt.plot(x,y)     
##    for i in border2:
##        x.append(i[1])
##        y.append(i[0])
##    plt.plot(x,y) 
#    x=[]
#    y=[]
#    xx=[]
#    yy=[]
#
#  
#   
##    for i,p in border1,borderline:
##    border1=border1[0:100]
##    borderline=borderline[0:100]
###################################################    
#    for (a, b) in zip(border1c,borderlinec):        
#        y.append(a[0])
#        y.append(b[0])
#        x.append(a[1])
#        x.append(b[1])
#        xx.append(x)
#        yy.append(y)
#        plt.plot(x,y) 
#        x=[]
#        y=[]
##################################################        
#    plt.savefig(pathdistance+"onlynodules"+fname, dpi=my_dpi,transparent=True)
#    plt.show()
    #include code for multiple nodules
    cv2.imwrite(pathdistance+"onlynodules_"+fname, imag) 
    return borderlinec,border1c,imag
#        print("hello")
#    print()        
    
def drawConsecBordersB2BOnlyNodules(border1,border2,borderline,E2tnf,fname,patid):
#    border1c=copy.copy(border1) 
##    border2c=copy.copy(border2)
#    borderlinec=copy.copy(borderline)
    pathdistance='./outputnew/'+patid+'/a2p/onlynodules/'
    my_dpi=96
    E2tnfc=copy.copy(E2tnf)
    border1c=[]
    borderlinec=[]
    for i,c in enumerate(E2tnf):
        if(E2tnf[i]==True):
            border1c.append(border1[i])
            borderlinec.append(borderline[i])
            
#        x.append(i[1])
#        y.append(i[0])    
#    border1c[~E2tnfc]=(0,0)
#    borderlinec[~E2tnfc]=(0,0)
##    border2[~E2tnf]=(0,0)    
    imag=np.zeros((512,512))
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in border1c:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
#    xy = [(a, b) for a in x for b in y]
    x=[]
    y=[]
########################################    
#    for i in borderlinec:
#        x.append(i[1])
#        y.append(i[0])
#    plt.plot(x,y)     
##    for i in border2:
##        x.append(i[1])
##        y.append(i[0])
##    plt.plot(x,y) 
#    x=[]
#    y=[]
#    xx=[]
#    yy=[]
#
#  
#   
##    for i,p in border1,borderline:
##    border1=border1[0:100]
##    borderline=borderline[0:100]
###################################################    
#    for (a, b) in zip(border1c,borderlinec):        
#        y.append(a[0])
#        y.append(b[0])
#        x.append(a[1])
#        x.append(b[1])
#        xx.append(x)
#        yy.append(y)
#        plt.plot(x,y) 
#        x=[]
#        y=[]
##################################################        
    plt.savefig(pathdistance+"onlynodules"+fname, dpi=my_dpi,transparent=True)
#    plt.show()
    #include code for multiple nodules
    return borderlinec,border1c
#        print("hello")
#    print()    
    
def drawConsecBordersOpenCV(border1,border2,fname,slice1,patid):
    imag=copy.copy(slice1)
    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)
#    imag=np.zeros((512,512,3))
#    imag1=copy.copy(imag)
    for pt in border1:
        imag[pt]=[0,0,255]
    for pt in border2:
        imag[pt]=[255,0,0]
    
#    for pt in border1:
##        imag1[pt]=[0,0,255]  
#        imag[pt]=[0,0,255]
##    cv2.imwrite("One"+fname, imag1)    
##    imag2=np.zeros((512,512,3))
#    for pt in border2:
##        imag2[pt]=[255,0,0] 
#        imag[pt]=[255,0,0]
#    cv2.imwrite("Two"+fname, imag2) 
    pathboth='./outputnew/'+patid+'/a2p/both/'        
    cv2.imwrite(pathboth+"Bth_1sd_DCBO_"+fname, imag) 

def drawConsecBorders(border1,border2,slice1,slice2):
#    imag=np.zeros((512,512))
    imag=slice1
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in border1:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
    x=[]
    y=[]    
    for i in border2:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
       
#    plt.plot(i[1],i[0])        
#    plt.plot([i[1] for i in border1], [i[0] for i in border1])  
#    plt.plot([i[1] for i in border2], [i[0] for i in border2]) 

def drawConsecBordersandLine(border1,border2,borderline):
    imag=np.zeros((512,512))
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    plt.plot([i[1] for i in border1], [i[0] for i in border1])  
    plt.plot([i[1] for i in border2], [i[0] for i in border2]) 
    plt.plot([i[1] for i in borderline], [i[0] for i in borderline], 'k-')

def drawConsecBordersandLineB2B(border1,border2,borderline):
    imag=np.zeros((512,512))
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    plt.plot([i[1] for i in border1], [i[0] for i in border1])  
    plt.plot([i[1] for i in border2], [i[0] for i in border2]) 
    plt.plot([i[1] for i in border1], [i[0] for i in borderline], 'k-')
#def drawConsecBordersandLineB2BOnlyNodules(border1,border2,borderline,E2tnf):
#    imag=np.zeros((512,512))
#    plt.figure()
#    plt.imshow(imag, cmap='Greys')
#    border1c=copy.copy(border1)
#    border2c=copy.copy(border2)
#    borderlinec=copy.copy(borderline)
#    E2tnfc=copy.copy(E2tnf)
#    border1c[~E2tnfc]=(0,0)
#    borderlinec[~E2tnfc]=(0,0)
##    border2[~E2tnf]=(0,0)
#    
##    plt.plot([i[1] for i in border1], [i[0] for i in border1])  
##    plt.plot([i[1] for i in border2], [i[0] for i in border2]) 
#    plt.plot([i[1] for i in border1c], [i[0] for i in borderlinec], 'k-')

def divideBorder(border):
    #for count,i in enumerate(border):
   # first_ten_lines = islice(border, 675)
    divborder=[]
    for i in islice(border, 1, len(border), 5): 
        divborder.append(i) 
    return divborder
#    for line in first_ten_lines:
#        print(line)
#    for count in islice(border, 2, None):
#        print(count)
def getBlockChainAsList(contours):
    biggest=0
    for i,j in enumerate(contours):
        shapej=j.shape[0]
        if(shapej>contours[biggest].shape[0]):
            biggest=i
    xi=[]
    for i in range(len(contours[biggest])):
        xi.append(contours[biggest][i][0][0])  
    yi=[]
    for i in range(len(contours[biggest])):
        yi.append(contours[biggest][i][0][1])  
    coordinates = zip(yi, xi) 
    blockchain=list(coordinates)
    return biggest, blockchain

def getlobeborders(image):
    plt.figure(),plt.imshow(image, cmap=plt.cm.gray)#, plt.show()
    labelled1 = label(image)
    regprop1=regionprops(labelled1)
    alllabels = [r.label for r in regionprops(labelled1)]
    areas = [r.area for r in regionprops(labelled1)]
    centroids=[r.centroid for r in regionprops(labelled1)]
    areas.sort()
    firstmaxarea=areas[0]
    secondmaxarea=areas[1]
    lab1posn=[i for i, j in enumerate(areas) if j == firstmaxarea]
    lab2posn=[i for i, j in enumerate(areas) if j == secondmaxarea]
    lab1centroid=centroids[lab1posn[0]]
    lab2centroid=centroids[lab2posn[0]]
    lab1posn=lab1posn[0]+1
    lab2posn=lab2posn[0]+1
    
    firstlobe=(labelled1==lab1posn)*255
    secondlobe=(labelled1==lab2posn)*255
    
    firstlobe=firstlobe.astype(np.uint8)
    secondlobe=secondlobe.astype(np.uint8)

    edges1, hierarchy = cv2.findContours(firstlobe, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index,blockchain1=getBlockChainAsList(edges1) 
    
#    cv2.drawContours(imag, edges1, -1, (0, 255, 0), 1) 

    edges2, hierarchy = cv2.findContours(secondlobe, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index,blockchain2=getBlockChainAsList(edges2) 
    
    if(lab1centroid[1]<lab2centroid[1]):
        
        return blockchain1,blockchain2,firstlobe,secondlobe
    else:
        return blockchain2,blockchain1,secondlobe,firstlobe   
    
##    print(lab1posn)
##    print(lab2posn)
##    firstlobeg = cv2.cvtColor(firstlobe,cv2.COLOR_BGR2GRAY)
##    plt.figure(),plt.imshow(firstlobe, cmap=plt.cm.gray)
#
##    xi=[edges1[biggest][i][0][0] ]
##    yi=[edges1[biggest][i][0][1] for i in range(len(edges1[biggest]))]
##    coordinates = zip(xi, yi)           
##    imag=np.zeros((512,512,3))
##    edges1.pop(0)
#    
#    
#    
##    cv2.imshow('dst_rt1', imag)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()
##    plt.figure(),plt.imshow(imag)
##    im1=imag[:,:,1]
##    plt.figure(),plt.imshow(im1, cmap=plt.cm.gray)
##    blockchain1=getChain(im1) 
#    
##    firstlobeg = cv2.cvtColor(firstlobe,cv2.COLOR_BGR2GRAY)
#    edges2, hierarchy = cv2.findContours(secondlobe, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    imag=np.zeros((512,512,3))
#    cv2.drawContours(imag, edges2, -1, (0, 255, 0), 1) 
##    cv2.imshow('dst_rt2', imag)
##    cv2.waitKey(0)
##    cv2.destroyAllWindows()    
#    im2=imag[:,:,1]
##    plt.figure(),plt.imshow(im2, cmap=plt.cm.gray)
#    blockchain2=getChain(im2) 
#
#    
##    edges1 = cv2.Canny(firstlobe, 100, 150, apertureSize=3)
##    imag1=copy.copy(edges1)
##    plt.figure(),plt.imshow(edges1, cmap=plt.cm.gray)
##    blockchain1=getChain(imag1)    
#     
##    edges2 = cv2.Canny(secondlobe, 100, 150, apertureSize=3)
##    imag2=copy.copy(edges2)
##    blockchain2=getChain(imag2)
#    if(lab1centroid[1]<lab2centroid[1]):
#        
#        return blockchain1,blockchain2,firstlobe,secondlobe
#    else:
#        return blockchain2,blockchain1,secondlobe,firstlobe

def getlobeborder(rightorleft,image):
    labelled1 = label(image)
    regprop1=regionprops(labelled1)
    alllabels = [r.label for r in regionprops(labelled1)]
    areas = [r.area for r in regionprops(labelled1)]
    centroids=[r.centroid for r in regionprops(labelled1)]
    areas.sort()
    firstmaxarea=areas[0]
    secondmaxarea=areas[1]
    lab1posn=[i for i, j in enumerate(areas) if j == firstmaxarea]
    lab2posn=[i for i, j in enumerate(areas) if j == secondmaxarea]
    lab1centroid=centroids[lab1posn[0]]
    lab2centroid=centroids[lab2posn[0]]
    lab1posn=lab1posn[0]+1
    lab2posn=lab2posn[0]+1
    
    firstlobe=(labelled1==lab1posn)*255
    secondlobe=(labelled1==lab2posn)*255
    
    firstlobe=firstlobe.astype(np.uint8)
    secondlobe=secondlobe.astype(np.uint8)
    
#    print(lab1posn)
#    print(lab2posn)
    
    edges1 = cv2.Canny(firstlobe, 100, 150, apertureSize=3)
    imag1=copy.copy(edges1)
    plt.figure(),plt.imshow(imagetempp, cmap=plt.cm.gray)
    blockchain1=getChain(imag1)    
     
    edges2 = cv2.Canny(secondlobe, 100, 150, apertureSize=3)
    imag2=copy.copy(edges2)
    blockchain2=getChain(imag2)
    return blockchain1,blockchain2
    



def createChain(borderl1,borderl2,fname,patid):
    pathdistance='./outputnew/'+patid+'/a2p/createchain/'

    newlobel1=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in borderl1:
        newlobel1[pt]=255 
    newlobel2=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in borderl2:
        newlobel2[pt]=255   
    blockchain1gc=getChain(newlobel1) 
    blockchain2gc=getChain(newlobel2) 
#    blockchain1,_ = cv2.findContours(newlobel1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    blockchain2,_ = cv2.findContours(newlobel2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blockchain1,_ = cv2.findContours(newlobel1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    blockchain2,_ = cv2.findContours(newlobel2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cv2.imwrite(pathdistance+"L1:CreateChain_b4"+fname, newlobel1) 
    cv2.imwrite(pathdistance+"L2:CreateChain_b4"+fname, newlobel2)    
    index1,blockchain1=getBlockChainAsList(blockchain1) 
    index2,blockchain2=getBlockChainAsList(blockchain2)      
#    blockchain1=getChain(newlobel1) 
#    blockchain2=getChain(newlobel2)
    newlobel1=np.zeros((512,512), dtype=np.uint8)
# #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
 #   blockchain1=blockchain1[0:449]
    blockchain1=Remove(blockchain1)
    for pt in blockchain1:
        newlobel1[pt]=255 
    newlobel2=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    blockchain2=Remove(blockchain2)   
    for pt in blockchain2:
        newlobel2[pt]=255    
    newlobel1gc=np.zeros((512,512), dtype=np.uint8)
    newlobel2gc=np.zeros((512,512), dtype=np.uint8)
    for pt in blockchain1gc:
        newlobel1gc[pt]=255 
  #  newlobel2=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in blockchain2gc:
        newlobel2gc[pt]=255  


    cv2.imwrite(pathdistance+"L1:CreateChain"+fname, newlobel1) 
    cv2.imwrite(pathdistance+"L2:CreateChain"+fname, newlobel2) 

    cv2.imwrite(pathdistance+"L1:CreateChain_gc"+fname, newlobel1gc) 
    cv2.imwrite(pathdistance+"L2:CreateChain_gc"+fname, newlobel2gc)     
    return blockchain1,blockchain2  

        
#    plt.figure(), plt.imshow(newlobel1, cmap=plt.cm.gray)#, plt.show()
#    plt.figure(), plt.imshow(newlobel2, cmap=plt.cm.gray)#, plt.show()
#    labelled1 = label(newlobel1)
#    regprop1=regionprops(newlobel1)
#    labelled2 = label(newlobel2)
#    regprop2=regionprops(newlobel2)    
#    
#    alllabels1 = [r.label for r in regionprops(labelled1)]
#    areas1 = [r.area for r in regionprops(labelled1)]
#    centroids1=[r.centroid for r in regionprops(labelled1)]
#    areas1.sort()
#    alllabels2 = [r.label for r in regionprops(labelled2)]
#    areas2 = [r.area for r in regionprops(labelled2)]
#    centroids2=[r.centroid for r in regionprops(labelled2)]
#    areas2.sort()
#    
#    maxarea1=areas1[0]
#    maxarea2=areas2[0]
#
#    lab1posn=[i for i, j in enumerate(areas1) if j == maxarea1]
#    lab2posn=[i for i, j in enumerate(areas2) if j == maxarea2]
#    
#    
##    lab1centroid=centroids[lab1posn[0]]
##    lab2centroid=centroids[lab2posn[0]]
#    lab1posn=lab1posn[0]+1
#    lab2posn=lab2posn[0]+1
#    
#    lobe1=(labelled1==lab1posn)*255
#    lobe2=(labelled2==lab2posn)*255
#    
#    lobe1=lobe1.astype(np.uint8)
#    lobe2=lobe2.astype(np.uint8)
##
#    edges1, hierarchy = cv2.findContours(lobe1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    index1,blockchain1=getBlockChainAsList(edges1) 
#    edges2, hierarchy = cv2.findContours(lobe2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    index2,blockchain2=getBlockChainAsList(edges2)     
#    newlobel1=np.zeros((512,512), dtype=np.uint8)
# #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
#    for pt in blockchain1:
#        newlobel1[pt]=255 
#    newlobel2=np.zeros((512,512), dtype=np.uint8)
# #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
#    for pt in blockchain2:
#        newlobel2[pt]=255 
#    cv2.imwrite(pathdistance+"L1:RemovePieces"+fname, newlobel1) 
#    cv2.imwrite(pathdistance+"L2:RemovePieces"+fname, newlobel2) 
#    return blockchain1,blockchain2   




def getImgFormatted(img):
    imagetempp=copy.copy(img)
    imagetempp[img==0]=255
    imagetempp[img==1]=0
    return imagetempp
#    plt.figure(),plt.imshow(imagetempp, cmap=plt.cm.gray)   
def selectNborders(border,n):
    borderpoints=border[0::n]
#    data = [1, 2, "|", "|", 3, 4]
#    [list(g) for k, g in groupby(data, key=lambda x: x != "|") if k]    
#    border1=borderpoints
    return borderpoints
def select2bordersGroupsTwo(border,n):
    borderpoints=border[0::n]
#    data = [1, 2, "|", "|", 3, 4]
    midpoint=borderpoints[1]
    grouppoints=[list(g) for k, g in groupby(border, key=lambda x: x != midpoint) if k] 
#    test_list = [6] + test_list 
    grouppoints[1]=[midpoint]+grouppoints[1]
#    grouppoints[0].append(grouppoints[1][0])
#    border1=borderpoints
    return grouppoints,borderpoints
def drawBorderLines(border):
    x=[ x[0] for x in border]
    y=[ x[1] for x in border]
    plt.figure()
    plt.plot(y,x, 'k-')
    plt.ylim(512, 0)
    plt.xlim(0, 512)
    plt.axis('equal')
 #   plt.show()
    
def distance(x1 , y1 , x2 , y2): 
  
    # Calculating distance 
    return math.sqrt(math.pow(x2 - x1, 2) +
                math.pow(y2 - y1, 2) * 1.0) 
  
# Drivers Code 
#print("%.6f"%distance(3, 4, 4, 3)) 
#def findCorrespondingInOtherBorder(borderpoint,border):
#    first=borderpoint[0]
#    second=borderpoint[1]
#    xcount=[]
##    ycount=[]
#    for x in enumerate(border): 
#        if (x[1][0]==first):
#            xcount.append(x)
#    for x in enumerate(border): 
#        if (x[1][1]==second):
#            xcount.append(x)   
#                
##    xcount=[x[0] for x in enumerate(border) if x[0]==first]
##    ycount=[x[0] for x in enumerate(border) if x[0]==second]
#    return xcount

def findCorrespondingInOtherBorderShortest(borderpoint,border):
    first=borderpoint[0]
    second=borderpoint[1]
    xcount=[]
    ycount=[]
    shortestdistance=1000
    dist1=1000
    dist2=1000
    
    for x in enumerate(border): 
        if (x[1][0]==first):
            dist=distance(first,second,x[1][0],x[1][1])
            if(dist<shortestdistance):
                shortestdistance=dist
                xcount=x
#                xcount.append(x)
    for x in enumerate(border): 
        if (x[1][1]==second):
            dist=distance(first,second,x[1][0],x[1][1])
            if(dist<shortestdistance):
                shortestdistance=dist  
                ycount=x
#                ycount.append(x)            
    if(len(xcount)>0):
        dist1=distance(first,second,xcount[1][0],xcount[1][1])
    if(len(ycount)>0):        
        dist2=distance(first,second,ycount[1][0],ycount[1][1])
    if(dist1>dist2):
        return 1,ycount
    else:
        return 0,xcount
                    
#    xcount=[x[0] for x in enumerate(border) if x[0]==first]
#    ycount=[x[0] for x in enumerate(border) if x[0]==second]
#    return xcount,ycount

#def findCorrespondingInOtherBorder(borderpoint,border):
#    first=borderpoint[0]
#    second=borderpoint[1]
#    xcount=[]
#    ycount=[]
#    for x in enumerate(border): 
#        if (x[1][0]==first):
#            xcount.append(x)
#    for x in enumerate(border): 
#        if (x[1][1]==second):
#            ycount.append(x)            
##    xcount=[x[0] for x in enumerate(border) if x[0]==first]
##    ycount=[x[0] for x in enumerate(border) if x[0]==second]
#    return xcount,ycount
    
#    xcount=[ x[0]==first for x in borderl2]
#    ycount=[ x[0]==second for x in borderl2]
#    
        
def findMiddleslice(lsvolsh):
    volsh=lsvolsh.shape
    area=[]
    for i in range(volsh[2]):
        img=lsvolsh[:,:,i]
#        hull = ConvexHull(img)
#        hull=img
        ar=np.sum(img)
        area.append(ar)
    difference=[]
    #diff=0
    
    for i in range(0,len(area)-1):
#    enumerate(area):
#        if(i<len(area)-2):
#        print(i)
        diff=area[i+1]-area[i]
        diff=np.int64(diff)
#        diff=int64(diff)
        difference.append(diff)
    #plt.plot(x, y, 'o', color='black');  
    x=[i for i in range(len(difference))]
    y=difference
#    plt.figure()
#    #plt.plot(x, y, 'o', color='black');
#    plt.style.use('seaborn-whitegrid')
#    plt.plot(x, y, '-ok');
#    plt.show()      
#    print("hello")   
    
    
    #x1 = np.linspace(-3, 3, 50)
    #y1 = np.exp(-x1**2) + 0.1 * np.random.randn(50)
    
    x1=np.array(x,np.float64)
    y1=np.array(y,np.float64)
    
    plt.figure()
    plt.plot(x1, y1, 'ro', ms=5)
    x_smooth=np.linspace(x1.min(),x1.max(),len(x1))
    s=len(x1)*np.var(y1)
    #https://stackoverflow.com/questions/8719754/scipy-interpolate-univariatespline-not-smoothing-regardless-of-parameters
    smooth=UnivariateSpline(x1,y1,s=269346633)
    #smooth=NearestNDInterpolator(x1,y1)
    #y_smooth.set_smoothing_factor(5.5)
    ynew=smooth(x_smooth)
    plt.plot(x_smooth,smooth(x_smooth), 'g', lw=3)
    
#    plt.show()
    
    y_min=min(ynew)
    y_max=max(ynew)
    minslice=getSlice(y_min,y1,x1,s)
    maxslice=getSlice(y_max,y1,x1,s)
    minslice=int(minslice[0])
    maxslice=int(maxslice[0])
    yToFind = 0
    jkk=getSlice(yToFind,y1,x1,s)
#    yreduced = np.array(y1) - yToFind
#    freduced = UnivariateSpline(x1, yreduced, s=269346633)
#    jkk=freduced.roots()
    
    if(len(jkk)==1):
        midslice=jkk[0]
    elif(len(jkk)==2): 
        midslice=jkk[1]
    elif(len(jkk)==3):
        midslice=jkk[1]  
    else:
        midslice=jkk[1] 
    midslice=int(midslice)

    
    if (minslice>maxslice):
        temp=minslice
        minslice=maxslice
        maxslice=temp
    
        
    return midslice,minslice,maxslice       

def getSlice(yvalue,y1,x1,s):
    yToFind = yvalue
    yreduced = np.array(y1) - yToFind
    freduced = UnivariateSpline(x1, yreduced, s=269346633)
    jkk=freduced.roots() 
    return jkk
def getMin(x, y):
    f = interpolate.interp1d(x, y, kind="quadratic")
    xmin = optimize.fmin(lambda x: f(x), x[1])
    ymin = f(xmin)
    return xmin[0], ymin[0]    

       
def pairwiseDistancesArgMinMin(borderl1,borderl2):
    exp_data= np.asarray(borderl1)#array("t", borderl1)
    num_data= np.asarray(borderl2)#array("t", borderl1) 
    D, E = pairwise_distances_argmin_min(exp_data, num_data, metric="euclidean")
    alllabels=[]
#    for i,r in enumerate(D):
#        alllab=borderl2[i]
#        alllabels.append(allab)
    
    for i,r in enumerate(D):
        alllabels.append(borderl2[r])    
#    alllabels = [borderl2[i] for i,r in enumerate(D)]
    return D,E,alllabels

def plotImgHistogram(image,binary,thresh):
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes.ravel()
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
    
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    ax[1].hist(image.ravel(), bins=256)
    ax[1].set_title('Histogram')
    print("threshold    "+str(thresh))
    ax[1].axvline(thresh, color='r')
    
    ax[2].imshow(binary, cmap=plt.cm.gray)
    ax[2].set_title('Thresholded')
    ax[2].axis('off')
    
#    plt.show()

def segmentLung(im):
    global df
   # binary11 = im #    Step 0: Original Image.    
   # plt.figure(),plt.imshow(binary11, cmap=plt.cm.gray) 
   
#    imn=np.array(im)
#    imn.astype(np.int16)

    imtem=copy.copy(im)
#    minimum=min(im.flatten())
    get_high_vals = imtem <0# minimum
    imtem[get_high_vals] = 0
    imtemp=copy.copy(imtem)
    #imtem=copy.copy(imtemp)
    thresh1 = threshold_otsu(imtem)
    otsu = imtemp > thresh1
#    plotImgHistogram(imtemp,otsu,thresh1)
    labels,numlabel=measure.label(otsu, neighbors=None, background=None, return_num=True, connectivity=None)  #output is a int64 array
    negotsu=~otsu
    labels,numneglabel=measure.label(negotsu, neighbors=None, background=None, return_num=True, connectivity=None)  #output is a int64 array
#    print(numlabel)
#    print(numneglabel)   
    if(numlabel>100 and numneglabel>100):
#        print(numlabel)
#        print(numneglabel)
        selem = disk(5)
        medfiltered=median(otsu, selem)
    else:
        medfiltered=copy.copy(otsu)    
    
    
#    featurevector = [] 
#    featurevector.append(printstr)
#    featurevector.append(resultsubstring)
#    featurevector.append(numlabel)
#    featurevector.append(numneglabel)
#    df_temp = pd.DataFrame([featurevector],columns=names)
#    df=df.append(df_temp)
    
    
    
    
    
    

#    valuepsnr=psnr(otsu,medfiltered)
#    print(valuepsnr)
   # medfiltered=otsu
#    cleared1 = clear_border(otsu) #    Step 2: Remove the blobs connected to the border of the image.
    selem = disk(5) #    Step 5: Erosion operation with a disk of radius 2. This operation is 
#    cleared1=otsu
    eroded1 = binary_erosion(medfiltered, selem) #    seperate the lung nodules attached to the blood vessels.
    eroded1=medfiltered
#    labelled1 = label(eroded1)
#    regprop1=regionprops(labelled1)

    
#    areas = [r.area for r in regionprops(labelled1)]
#    if(len(areas) >=5):
#        last5areas=areas[-5:] 
#        maxv = max(last5areas)
#        maxposn=[i for i, j in enumerate(areas) if j == maxv]
#        maxarea=regprop1[maxposn[0]].area
#        erodip[regprop1[maxposn[0]].coords[:,0],regprop1[maxposn[0]].coords[:,1]]=False
#        regpropped1=erodip 
#    else:
#        regpropped1=erodip    
 #   print(maxarea)
  #  regprop1[maxposn[0]].coords[0]=False
   
    #regprop1 is to remove the bottom line from the image
    #regprop2 is to segment only the region outside the parenchyma (it is the white region with largest area)
  #  eroded1=np.pad(eroded1,[2, 2], 'constant',constant_values=True)
    eroded1[0,:] = eroded1[:,0] = eroded1[:,-1] =  eroded1[-1,:] = False
    erodip=copy.copy(eroded1)
    regpropped2=copy.copy(erodip)
    labelled2 = label(regpropped2)
    regprop2=regionprops(labelled2)
    areas = [r.area for r in regionprops(labelled2)]
#    plotImgHistogram(imtemp,eroded1,thresh1)
    maxv = max(areas)
    maxposn=[i for i, j in enumerate(areas) if j == maxv]
    #maxarea=regprop2[maxposn[0]].area
  #  regpropped2[regprop2[maxposn[0]].coords[:,0],regprop2[maxposn[0]].coords[:,1]]=False
    
    #take a fresh 2d image and plot the segmented large region outside parenchyma in tempim
    
    tempim=np.zeros((512,512), dtype=np.bool)
    tempim[regprop2[maxposn[0]].coords[:,0],regprop2[maxposn[0]].coords[:,1]]=True
    
    #negtempim consists of two regions one outside the body and inside the parenchyma
    #regprop3 is to segment the parenchyma region
    negtempim=~ tempim
    regpropped3=copy.copy(negtempim)
    labelled3 = label(regpropped3)
    regprop3=regionprops(labelled3)
    labels = [r.label for r in regionprops(labelled3)]
   # maxv = max(areas)
    
    lab1posn=[i for i, j in enumerate(labels) if j == 1]
  #  maxarea=regprop3[maxposn[0]].area
    
    regpropped3[regprop3[lab1posn[0]].coords[:,0],regprop3[lab1posn[0]].coords[:,1]]=False
    labelled4 = label(regpropped3)
    regprop4=regionprops(labelled4)
    areas = [r.area for r in regionprops(labelled4)]
    areascp=copy.copy(areas)
    areascp.sort(reverse = True)
    if(len(areascp)>=2):
        del areascp[2:]
        maxv1 = areascp[0]
        maxposn1=[i for i, j in enumerate(areas) if j == maxv1]
        tempim=np.zeros((512,512), dtype=np.bool)
        tempim[regprop4[maxposn1[0]].coords[:,0],regprop4[maxposn1[0]].coords[:,1]]=True
        maxv2 = areascp[1]
        maxposn2=[i for i, j in enumerate(areas) if j == maxv2]
    #    tempim=np.zeros((512,512), dtype=np.bool)
        tempim[regprop4[maxposn2[0]].coords[:,0],regprop4[maxposn2[0]].coords[:,1]]=True    
    else:
        tempim=np.zeros((512,512), dtype=np.bool)
#    selem = disk(2) #    Step 5: Erosion operation with a disk of radius 2. This operation is 
#    #regpropped3in=copy.copy(regpropped3)
#    eroded2 = binary_erosion(regpropped3, selem)
#    
    

#    selem = disk(10)#    Step 6: Closure operation with a disk of radius 10. This operation is
#    closed1 = binary_closing(eroded1, selem) #    to keep nodules attached to the lung wall.   

    

#    closed1=regpropped3

    
 #   areas = [r.area for r in regionprops(label_image)]
    
#    edges = roberts(closed1)#    Step  7: Fill in the small holes inside the binary mask of lungs.
#    filled1 = ndimage.binary_fill_holes(edges)  
 #   selem = disk(1) #    Step 5: Erosion operation with a disk of radius 2. This operation is 
#    eroded2 = binary_erosion(filled1, selem) #    seperate the lung nodules attached to the blood vessels.
#    get_high_vals = regpropped3 == 0 #    Step  8: Superimpose the binary mask on the input image.
#    
#    
#    imtemp[get_high_vals] = 0
    #plotImgHistogram(im,binary,thresh1)
    
#    plt.figure(),plt.imshow(im, cmap=plt.cm.gray)  
#    plt.title("original")   
#    plt.figure(),plt.imshow(imtem, cmap=plt.cm.gray)
#    plt.title("imtemp")    
#    plt.figure(),plt.imshow(otsu, cmap=plt.cm.gray)  
#    plt.title("otsu thresholded")    
#    plt.figure(),plt.imshow(medfiltered, cmap=plt.cm.gray)  
#    plt.title("medfiltered1")  
#    
#    
#    
#    plt.figure(),plt.imshow(eroded1, cmap=plt.cm.gray)  
#    plt.title("eroded1")
# #   plt.figure(),plt.imshow(regpropped1, cmap=plt.cm.gray)
##    plt.title("regpropped1")    
#    plt.figure(),plt.imshow(negtempim, cmap=plt.cm.gray)
#    plt.title("regpropped2")     
#    plt.figure(),plt.imshow(regpropped3, cmap=plt.cm.gray)
#    plt.title("regpropped3")     
#    plt.figure(),plt.imshow(tempim, cmap=plt.cm.gray)
#    plt.title("regpropped4")        
##    plt.figure(),plt.imshow(closed1, cmap=plt.cm.gray)
##    plt.title("closed1")
##    plt.figure(),plt.imshow(edges, cmap=plt.cm.gray)
##    plt.title("edges")    
# #   plt.figure(),plt.imshow(filled1, cmap=plt.cm.gray)  
##    plt.title("filled1")
##    plt.figure(),plt.imshow(eroded2, cmap=plt.cm.gray)  
##    plt.title("eroded2")    
##    im[get_high_vals] = 0
# #   plt.figure(),plt.imshow(imtemp, cmap=plt.cm.gray)   
##    plt.title("final")
# #   plt.figure(),plt.imshow(tempim, cmap=plt.cm.gray)   
##    plt.title("final")
# #   plt.figure(),plt.imshow(regpropped3, cmap=plt.cm.gray)   
# #   plt.title("final")
    return tempim

from PIL import Image
from PIL import ImageDraw

def make_bezier(xys):
    # xys should be a sequence of 2-tuples (Bezier control points)
    n = len(xys)
    combinations = pascal_row(n-1)
    def bezier(ts):
        # This uses the generalized formula for bezier curves
        # http://en.wikipedia.org/wiki/B%C3%A9zier_curve#Generalization
        result = []
        for t in ts:
            tpowers = (t**i for i in range(n))
            upowers = reversed([(1-t)**i for i in range(n)])
            coefs = [c*a*b for c, a, b in zip(combinations, tpowers, upowers)]
            result.append(
                tuple(sum([coef*p for coef, p in zip(coefs, ps)]) for ps in zip(*xys)))
        return result
    return bezier

def pascal_row(n, memo={}):
    # This returns the nth row of Pascal's Triangle
    if n in memo:
        return memo[n]
    result = [1]
    x, numerator = 1, n
    for denominator in range(1, n//2+1):
        # print(numerator,denominator,x)
        x *= numerator
        x /= denominator
        result.append(x)
        numerator -= 1
    if n&1 == 0:
        # n is even
        result.extend(reversed(result[:-1]))
    else:
        result.extend(reversed(result))
    memo[n] = result
    return result

def drawAllnearestborders(a):
    imag=np.zeros((512,512))
    plt.figure()
    plt.imshow(imag, cmap='Greys')
#    ts = [t for t in range(101)]
#    midaline=skimage.draw.line(a[0][0], a[0][1], a[1][0], a[1][1]) # 1. one type of border
#    skimage.draw.line(r0, c0, r1, c1) # 1. one type of border
 #   midacurve=skimage.draw.bezier_curve(a[0][0], a[0][1], a[1][0], a[1][1], r2, c2, weight, shape=None)
#    s
#    xys=a
#    bezier = make_bezier(xys)
#    points = bezier(ts)
    
    x=[]
    y=[]
    
    for ii in range(-1,1):
        for i in a:
            x.append(i[1]+ii)
            y.append(i[0]+ii)
        plt.plot(x,y) 
    plt.plot(midaline[0],midaline[1])    
 #   plt.show()
def drawNoduleContourAndNoduleNearestBorderContour(a,b,fname,patid):
    imag=np.zeros((512,512))
    pathdistance='./outputnew/'+patid+'/'
    my_dpi=96
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in a:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
   
    x=[]
    y=[]   
    for i in b:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y)  
    plt.savefig(pathdistance+"TESTING_FINAL"+fname, dpi=my_dpi,transparent=True)
#    plt.show()

def extraArmsForNodules(E2lst):
    E2lstc=copy.copy(E2lst)
    nostartlst=[]
    noendlst=[]
    prev=False
    for i,j in enumerate(E2lstc):
         if((j==True) & (prev==False)):
            nostartlst.append(i)
            prev=True
         if((j==True) & (prev==True)):
            prev=True
         if((j==False) & (prev==True)):
            noendlst.append(i-1) 
            prev=False 
    for i,j in enumerate(nostartlst):
         for ii in range(0,10):
             E2lstc[nostartlst[i]-ii]=True
             E2lstc[noendlst[i]+ii]=True
        
    return E2lstc            

#def convertCoordinatestoList():
#
#def convertListtoCoordinates():    
    
def checkBorder1TouchOverlapWithLobe3(a,lobe2):
    imag=np.zeros((512,512),dtype=np.uint8)

    x=[]
    y=[]
    alst=[]
    resultoverlaplst=[]
    resultarealst=[]
    for i in a:
        y.append(i[1])
        x.append(i[0])
            
    for ii in range(-3,3):
        freshalst=[]
        x=[]
        y=[]
        imag=np.zeros((512,512),dtype=np.uint8)
#        plt.figure()
#        plt.imshow(lobe2, cmap='Greys')
        for i in a:
            y.append(i[1]+ii)
            x.append(i[0]+ii)
        for (aa, bb) in zip(x,y):
            alst.append((aa,bb))
            freshalst.append((aa,bb))
        ar = np.array(freshalst)                  # Convert list to numpy array
        imag[ar[:,0], ar[:,1]] = 255    
        get_lobe_t_f=lobe2==255
        get_border_t_f=imag==255
        res_t_a_f=np.bitwise_and(get_lobe_t_f,get_border_t_f)
        res_t_o_f=np.bitwise_or(get_lobe_t_f,get_border_t_f)
        res_filled_image=ndimage.binary_fill_holes(res_t_o_f)#.astype(int)
        plt.figure(),plt.imshow(res_filled_image)#,plt.show()
        res_filled_image=res_filled_image*1
        area=np.sum(res_filled_image)
        res_t_a_f=res_t_a_f*1
        overlap=np.sum(res_t_a_f)
        resultarealst.append(area)
        resultoverlaplst.append(overlap)
#        plt.plot(y,x) 
#        plt.show()  
    print("LobeHello")

def checkBorder1TouchOverlapWithLobe2(a,lobe2):
    imag=np.zeros((512,512),dtype=np.uint8)

    x=[]
    y=[]
    alst=[]
    resultoverlaplst=[]
    resultarealst=[]
    for ii in range(-3,3):
        freshalst=[]
        x=[]
        y=[]
        imag=np.zeros((512,512),dtype=np.uint8)
#        plt.figure()
#        plt.imshow(lobe2, cmap='Greys')
        for i in a:
            y.append(i[1]+ii)
            x.append(i[0]+ii)
        for (aa, bb) in zip(x,y):
            alst.append((aa,bb))
            freshalst.append((aa,bb))
        ar = np.array(freshalst)                  # Convert list to numpy array
        imag[ar[:,0], ar[:,1]] = 255    
        get_lobe_t_f=lobe2==255
        get_border_t_f=imag==255
        res_t_a_f=np.bitwise_and(get_lobe_t_f,get_border_t_f)
        res_t_o_f=np.bitwise_or(get_lobe_t_f,get_border_t_f)
        res_filled_image=ndimage.binary_fill_holes(res_t_o_f)#.astype(int)
        plt.figure(),plt.imshow(res_filled_image)#,plt.show()
        res_filled_image=res_filled_image*1
        area=np.sum(res_filled_image)
        res_t_a_f=res_t_a_f*1
        overlap=np.sum(res_t_a_f)
        resultarealst.append(area)
        resultoverlaplst.append(overlap)
#        plt.plot(y,x) 
#        plt.show()  
    print("LobeHello")
#    return rightBorder

def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def getPointsFromTo(borderl1,slimnodulenearestborder1):
    posn=[]
    for i,j in enumerate(slimnodulenearestborder1):
        for ii, jj in enumerate(borderl1):
                if(j == jj):
                    posn.append(ii)       
    posn.sort()
    firstposn=posn[0]
    lastposn=posn[-1]
    newlist=borderl1[firstposn:lastposn]
    
#    firstpoint=slimnodulenearestborder1[0]
#    lastpoint=slimnodulenearestborder1[-1]
#
#    for i, j in enumerate(borderl1):
#        if(j == firstpoint):
#            lab1posn=i
#    for i, j in enumerate(borderl1):
#        if(j == lastpoint):
#            lab2posn=i   
#    if(lab1posn<lab2posn):
#        newlist=borderl1[lab1posn:lab2posn]
#    else:
#        newlist=borderl1[lab2posn:lab1posn]
        
    return newlist            
#    lab1posn=[i for i, j in enumerate(borderl1) if j == firstpoint]
#    lab2posn=[i for i, j in enumerate(areas) if j == secondmaxarea]
def checkLungbeginning(segmentedlung):
    labelled1 = label(segmentedlung)
    regprop1=regionprops(labelled1)
    alllabels = [r.label for r in regionprops(labelled1)]
    areas = [r.area for r in regionprops(labelled1)]
    ratio=[]
    for i,j in enumerate(alllabels):
        get_image=(labelled1==j)*255
        get_image=get_image.astype(np.uint8)
        edges1, hierarchy = cv2.findContours(get_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        meaedges1=len(edges1[0])
        ratio.append(meaedges1/areas[i])
    return ratio    
#    print("edge")    
    
    
#    alllabels = [r.label for r in regionprops(labelled1)]
#    areas = [r.area for r in regionprops(labelled1)]
#    centroids=[r.centroid for r in regionprops(labelled1)]
#    areas.sort()
#    firstmaxarea=areas[0]
#    secondmaxarea=areas[1]
#    lab1posn=[i for i, j in enumerate(areas) if j == firstmaxarea]
#    lab2posn=[i for i, j in enumerate(areas) if j == secondmaxarea]  

def AnteriorPosteriorDetection(lobel1,lobel2,lober1,lober2,borderl1,borderl2,borderr1,borderr2):
    labelledl1 = label(lobel1)
    labelledl2 = label(lobel2)
    labelledr1 = label(lober1)
    labelledr2 = label(lober2)          
    centroidl1=regionprops(labelledl1)[0].centroid
    centroidl2=regionprops(labelledl2)[0].centroid
    centroidr1=regionprops(labelledr1)[0].centroid
    centroidr2=regionprops(labelledr2)[0].centroid
    labposnl1=[]
    labposnl2=[]
    labposnr1=[]
    labposnr2=[]
    for i,j in enumerate(borderl1):
        if int(centroidl1[0])==j[0]:
                labposnl1.append(i)
    for i,j in enumerate(borderl2):
        if int(centroidl2[0])==j[0]:
                labposnl2.append(i)    
    for i,j in enumerate(borderr1):
        if int(centroidr1[0])==j[0]:
                labposnr1.append(i)
    for i,j in enumerate(borderr2):
        if int(centroidr2[0])==j[0]:
                labposnr2.append(i)  



def calculateDistanceBetweenAllPairsandReturnLongestDistIndex1(hullpts):
    cords=hullpts
    length=len(cords) 
    cords.append(cords[0])
    dist=[]   
    for i in range(length+1): 
        if (i==0):
            beg=cords[i]
            continue
        else:
            nxt=cords[i]
            dst = euclidean(beg,nxt)
            beg=nxt
            dist.append(dst)
    maxpos = dist.index(max(dist)) 
    
    if(maxpos+1==length):
        last=0
    else:
        last=maxpos+1 
    return maxpos,last,dist

def AnteriorPosteriorDetection2(lobel1,borderl1,slice,fname,patid):
    sliceimg= cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
    print("1")
  #  plt.figure(),plt.imshow(sliceimg)#,cmap=plt.cm.gray)
    global hull,longestdistancePointindices
#    lobel=[]
#    lober=[]
#    lobel.extend([lobel1,lobel2])
#    lober.extend([lober1,lober2])
    
#    lobes.append(lobe1,lobe)
    hullpts=[]
    borderl1c=copy.copy(borderl1)
    startindex=0
    endindex=0
#    borderl1c.append(borderl1c[0])

    pathlobe='./outputnew/'+patid+'/a2p/lobe/'
    pathhull='./outputnew/'+patid+'/a2p/hull/'
    pathnewborder='./outputnew/'+patid+'/a2p/newborder/'
    pathborder='./outputnew/'+patid+'/a2p/border/'

            
    plt.imsave(pathlobe+"lobe"+fname,lobel1,cmap='gray')
#    plt.figure(),plt.imshow(lobel1,cmap=plt.cm.gray)
#    plt.savefig("lobe"+fname)
    
    #imag=np.zeros((512,512,3))
    
    sliceimg1=copy.copy(sliceimg)
    for pt in borderl1:
        sliceimg1[pt]=[0,0,255]    
#    for pt in border2:
#        imag[pt]=[0,255,0]   
#    plt.imsave("border"+fname,imag,cmap='gray')
    filename="border"+fname
#    saveImageOpencv(filename,)
    cv2.imwrite(pathborder+"border"+fname,sliceimg1)
#    plt.figure(),plt.imshow(imag)
#    plt.savefig("border"+fname)        
        
#    plt.savefig(fname)
    contlist=[]
#    for i,j in enumerate(lobel1):
#        ret, thresh = cv2.threshold(j, 127, 255, 0)
        # Find the contours
    contours, hierarchy = cv2.findContours(lobel1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        del contours[1:]
    bigindex,listcontour=getBlockChainAsList(contours)
    contlist.append(contours[bigindex])
#        contours=contours[0]
        # For each contour, find the convex hull and draw it
        # on the original image.
    iii=0
#        for cnt in contours:
#            # some code in here
#            fn="cnt"+str(iii)+fname
#            iii=iii+1
#            cv2.imwrite(fn, cnt)
            #plt.imsave(fn,cnt,cmap='gray')
#            cv2.imwrite(fn, cnt)
#    for i in range(len(contlist)):
    #imag=np.zeros((512,512,3))
    sliceimg2=copy.copy(sliceimg)
    hull = cv2.convexHull(contlist[0])
#            hullshape=hull.shape
#            hull[hullshape[0]][0][0]=hull[0][0][0]
#            hull[hullshape[0]][0][1]=hull[0][0][1]
    coordinates = zip(hull[:,0,1], hull[:,0,0])
    pts=list(coordinates)
    hullpts=copy.copy(pts)
    pts.append(pts[0])
#    hullpts=copy.copy(pts)
    for pt in pts:
        sliceimg2[pt]=[0,255,0] 
    cv2.imwrite(pathhull+"hull"+fname, sliceimg2)

    
    
    #THE CONVEX HULL CAN BE DRAWN HERE
    longestdistancePointindex,nexttolongestdistancePointindex,dista=calculateDistanceBetweenAllPairsandReturnLongestDistIndex1(hullpts)
        
#        l=list1.index(find)
#        l=borderl1c.index(pts[longestdistancePointindices])
        
    for ii,jj in enumerate(borderl1c):
#                if((hull[longestdistancePointindices][0][0]==borderl1[ii][1]) & (hull[longestdistancePointindices][0][1]==borderl1[ii][0])):
        if((pts[longestdistancePointindices][1]==borderl1[ii][1]) & (pts[longestdistancePointindices][0]==borderl1[ii][0])):
            startindex=ii
            break
    for ii,jj in enumerate(borderl1c):
#                if((hull[longestdistancePointindices+1][0][0]==borderl1[ii][1]) & (hull[longestdistancePointindices+1][0][1]==borderl1[ii][0])):
        if((pts[longestdistancePointindices+1][1]==borderl1[ii][1]) &(pts[longestdistancePointindices+1][0]==borderl1[ii][0])):
            endindex=ii 
            break

    if(startindex>endindex):
        borderl1c1=borderl1c[startindex:]+borderl1c[0:endindex]
        borderl1c2=borderl1c[endindex:startindex]
    if(endindex>startindex):
        borderl1c1=borderl1c[endindex:]+borderl1c[0:startindex]
        borderl1c2=borderl1c[startindex:endindex]

#        del borderl1c1[startindex:endindex+1]
#        del borderl1c1[]
#        temp=startindex
#        startindex=endindex
#        endindex=temp  
    
    
#CHECK IF THE RIGHT BORDER HAS BEEN DELETED    
#    indices = [(startindex, endindex)]
#    borderl1c1=[borderl1c[s:e+1] for s,e in indices] 
#    borderl1c1=borderl1c1[0]   
#    
#    borderl1c2=copy.copy(borderl1c)
#    del borderl1c2[startindex:endindex+1]
#    
#    hullc=copy.copy(hull)
#    del hullpts[longestdistancePointindices]
#    del hullpts[longestdistancePointindices]
    
                
              
#    del borderl1c[startindex:endindex]
    
#    hullc=copy.copy(hull)
    
    intersectlist1=intersection(hullpts,borderl1c1)
    intersectlist2=intersection(hullpts,borderl1c2)
    if(len(intersectlist1)>len(intersectlist2)):
        borderl1c=borderl1c1
    else: 
        borderl1c=borderl1c2
        
#    if not intersectlist1:
#        borderl1c=borderl1c2
#    if  not intersectlist2:
#        borderl1c=borderl1c1
        
            
        
    
    imag=np.zeros((512,512,3))
    for pt in borderl1c:
        sliceimg[pt]=[0,0,255]    
#    for pt in border2:
#        imag[pt]=[0,255,0]   
#    plt.imsave("border"+fname,imag,cmap='gray')
    cv2.imwrite(pathnewborder+"newborder"+fname, sliceimg)

        
    return borderl1c    





def calculateDistanceBetweenAllPairsandReturnLongestDistIndex(hull):
    newrow = hull[0]
    newrow=newrow.reshape((1,1,2))
    hull = np.vstack([hull, newrow])
    P=[hull[0][0][0], hull[0][0][1]]
    distancebw=[]
    lab1posn=[]
    for i in range(1,len(hull)):
        Q=[hull[i][0][0], hull[i][0][1]]
        Distance=sqrt(sum((px - qx) ** 2.0 for px, qx in zip(P, Q)))
#        Distance = math.dist(P, Q) 
        distancebw.append(Distance)
        P=Q
    maxdist=max(distancebw)   
    for i, j in enumerate(distancebw):
        if j == max(distancebw):
            lab1posn.append(i)
    return lab1posn[0]        
#    lab1posn=[i for i, j in enumerate(distancebw) if j == max(distancebw)]
       
#    print("helloj")
#def getBlockChain(contours):
#    biggest=0
#    for i,j in enumerate(contours):
#        shapej=j.shape[0]
#        if(shapej>contours[biggest].shape[0]):
#            biggest=i
#    xi=[]
#    for i in range(len(contours[biggest])):
#        xi.append(contours[biggest][i][0][0])  
#    yi=[]
#    for i in range(len(contours[biggest])):
#        yi.append(contours[biggest][i][0][1])  
#    coordinates = zip(yi, xi) 
#    blockchain=list(coordinates)
#    return biggest, blockchain
def intersection(lst1, lst2): 
      
    return [item for item in lst1 if item in lst2] 

def AnteriorPosteriorDetection1(lobel1,borderl1,slice,fname,patid,LorR):
    global hull,longestdistancePointindices
    if (LorR):
        sliceimg= cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
        print("2")
    #    plt.figure(),plt.imshow(sliceimg)#,cmap=plt.cm.gray)
        pathlobe='./outputnew/'+patid+'/a2p/lobe/'
        pathhull='./outputnew/'+patid+'/a2p/hull/'
        pathnewborder='./outputnew/'+patid+'/a2p/newborder/'
        pathborder='./outputnew/'+patid+'/a2p/border/'
        
        hullpts=[]
        borderl1c=copy.copy(borderl1)
        startindex=0
        endindex=0               
    
        plt.imsave(pathlobe+"lobe"+fname,lobel1,cmap='gray')
        sliceimg1=copy.copy(sliceimg)
        for pt in borderl1:
            sliceimg1[pt]=[0,0,255]    
        filename="border"+fname
        cv2.imwrite(pathborder+"border"+fname,sliceimg1)
        contlist=[]
        contours, hierarchy = cv2.findContours(lobel1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bigindex,listcontour=getBlockChainAsList(contours)
        borderl1c=listcontour
        contlist.append(contours[bigindex])
        iii=0
        sliceimg2=copy.copy(sliceimg)
        hull = cv2.convexHull(contlist[0])
        coordinates = zip(hull[:,0,1], hull[:,0,0])
        pts=list(coordinates)
        hullpts=copy.copy(pts)
        pts.append(pts[0])
        hullpts=copy.copy(pts)
        for pt in pts:
            sliceimg2[pt]=[0,255,0] 
        cv2.imwrite(pathhull+"hull"+fname, sliceimg2)
        #THE CONVEX HULL CAN BE DRAWN HERE
        longestdistancePointindices=calculateDistanceBetweenAllPairsandReturnLongestDistIndex(hull)
        for ii,jj in enumerate(borderl1c):
            if((pts[longestdistancePointindices][1]==borderl1c[ii][1]) & (pts[longestdistancePointindices][0]==borderl1c[ii][0])):
                startindex=ii
                break
        for ii,jj in enumerate(borderl1c):
            if((pts[longestdistancePointindices+1][1]==borderl1c[ii][1]) &(pts[longestdistancePointindices+1][0]==borderl1c[ii][0])):
                endindex=ii 
                break
    
        if(startindex>endindex):
            temp=startindex
            startindex=endindex
            endindex=temp  
    
        lengthbord=len(borderl1c)
    #CHECK IF THE RIGHT BORDER HAS BEEN DELETED    
        indices = [(startindex, endindex)]
        borderl1c1=[borderl1c[s:e+1] for s,e in indices] 
        borderl1c1=borderl1c1[0]   
        borderl1c2=copy.copy(borderl1c)
        borderl1c2=borderl1c2[endindex:lengthbord]+borderl1c2[0:startindex]
        hullc=copy.copy(hull)
        del hullpts[longestdistancePointindices]
        del hullpts[longestdistancePointindices]
        intersectlist1=intersection(hullpts,borderl1c1)
        intersectlist2=intersection(hullpts,borderl1c2)
        if(len(intersectlist1)>len(intersectlist2)):
            borderl1c=borderl1c1
        else: 
            borderl1c=borderl1c2
        imag=np.zeros((512,512,3))
        for pt in borderl1c:
            sliceimg[pt]=[0,0,255]    
        cv2.imwrite(pathnewborder+"newborder"+fname, sliceimg)
        return borderl1c
    else:
        sliceimg= cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
        print("3")
    #    plt.figure(),plt.imshow(sliceimg)#,cmap=plt.cm.gray)
#        global hull,longestdistancePointindices
        pathlobe='./outputnew/'+patid+'/a2p/loberight/'
        pathhull='./outputnew/'+patid+'/a2p/hullright/'
        pathnewborder='./outputnew/'+patid+'/a2p/newborderright/'
        pathborder='./outputnew/'+patid+'/a2p/borderright/'
               
        hullpts=[]
        borderl1c=copy.copy(borderl1)
        startindex=0
        endindex=0
    
        plt.imsave(pathlobe+"lobe"+fname,lobel1,cmap='gray')
        sliceimg1=copy.copy(sliceimg)
        for pt in borderl1:
            sliceimg1[pt]=[0,0,255]    
        filename="border"+fname
        cv2.imwrite(pathborder+"border"+fname,sliceimg1)
        contlist=[]
        contours, hierarchy = cv2.findContours(lobel1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        bigindex,listcontour=getBlockChainAsList(contours)
        borderl1c=listcontour
        contlist.append(contours[bigindex])
        iii=0
    
        sliceimg2=copy.copy(sliceimg)
        hull = cv2.convexHull(contlist[0])
        coordinates = zip(hull[:,0,1], hull[:,0,0])
        pts=list(coordinates)
        hullpts=copy.copy(pts)
        pts.append(pts[0])
        hullpts=copy.copy(pts)
        for pt in pts:
            sliceimg2[pt]=[0,255,0] 
        cv2.imwrite(pathhull+"hull"+fname, sliceimg2)
        #THE CONVEX HULL CAN BE DRAWN HERE
        longestdistancePointindices=calculateDistanceBetweenAllPairsandReturnLongestDistIndex(hull)
        for ii,jj in enumerate(borderl1c):
            if((pts[longestdistancePointindices][1]==borderl1c[ii][1]) & (pts[longestdistancePointindices][0]==borderl1c[ii][0])):
                startindex=ii
                break
        for ii,jj in enumerate(borderl1c):
            if((pts[longestdistancePointindices+1][1]==borderl1c[ii][1]) &(pts[longestdistancePointindices+1][0]==borderl1c[ii][0])):
                endindex=ii 
                break
    
        if(startindex>endindex):
            temp=startindex
            startindex=endindex
            endindex=temp  
    
        lengthbord=len(borderl1c)
    #CHECK IF THE RIGHT BORDER HAS BEEN DELETED    
        indices = [(startindex, endindex)]
        borderl1c1=[borderl1c[s:e+1] for s,e in indices] 
        borderl1c1=borderl1c1[0]   
        borderl1c2=copy.copy(borderl1c)
       # borderl1c2=borderl1c2[startindex:endindex]
        borderl1c2=borderl1c2[endindex:lengthbord]+borderl1c2[0:startindex]
        hullc=copy.copy(hull)
        del hullpts[longestdistancePointindices]
        del hullpts[longestdistancePointindices]
        intersectlist1=intersection(hullpts,borderl1c1)
        intersectlist2=intersection(hullpts,borderl1c2)
        if(len(intersectlist1)>len(intersectlist2)):
            borderl1c=borderl1c1
        else: 
            borderl1c=borderl1c2
        imag=np.zeros((512,512,3))
        for pt in borderl1c:
            sliceimg[pt]=[0,0,255]    
        cv2.imwrite(pathnewborder+"newborder"+fname, sliceimg)
        return borderl1c        
    
def AnteriorPosteriorDetection1a(lobel1,borderl1,slice,fname,patid,LorR):
    sliceimg= cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
    print("4")
 #   plt.figure(),plt.imshow(sliceimg)#,cmap=plt.cm.gray)
    global hull,longestdistancePointindices
#    lobel=[]
#    lober=[]
#    lobel.extend([lobel1,lobel2])
#    lober.extend([lober1,lober2])
    
#    lobes.append(lobe1,lobe)
    hullpts=[]
    borderl1c=copy.copy(borderl1)
    startindex=0
    endindex=0
#    borderl1c.append(borderl1c[0])

    pathlobe='./outputnew/'+patid+'/a2p/lobe/'
    pathhull='./outputnew/'+patid+'/a2p/hull/'
    pathnewborder='./outputnew/'+patid+'/a2p/newborder/'
    pathborder='./outputnew/'+patid+'/a2p/border/'

            
    plt.imsave(pathlobe+"lobe"+fname,lobel1,cmap='gray')
#    plt.figure(),plt.imshow(lobel1,cmap=plt.cm.gray)
#    plt.savefig("lobe"+fname)
    
    #imag=np.zeros((512,512,3))
    
    sliceimg1=copy.copy(sliceimg)
    for pt in borderl1:
        sliceimg1[pt]=[0,0,255]    
#    for pt in border2:
#        imag[pt]=[0,255,0]   
#    plt.imsave("border"+fname,imag,cmap='gray')
    filename="border"+fname
#    saveImageOpencv(filename,)
    cv2.imwrite(pathborder+"border"+fname,sliceimg1)
#    plt.figure(),plt.imshow(imag)
#    plt.savefig("border"+fname)        
        
#    plt.savefig(fname)
    contlist=[]
#    for i,j in enumerate(lobel1):
#        ret, thresh = cv2.threshold(j, 127, 255, 0)
        # Find the contours
    contours, hierarchy = cv2.findContours(lobel1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        del contours[1:]
    bigindex,listcontour=getBlockChainAsList(contours)
    contlist.append(contours[bigindex])
#        contours=contours[0]
        # For each contour, find the convex hull and draw it
        # on the original image.
    iii=0
#        for cnt in contours:
#            # some code in here
#            fn="cnt"+str(iii)+fname
#            iii=iii+1
#            cv2.imwrite(fn, cnt)
            #plt.imsave(fn,cnt,cmap='gray')
#            cv2.imwrite(fn, cnt)
#    for i in range(len(contlist)):
    #imag=np.zeros((512,512,3))
    sliceimg2=copy.copy(sliceimg)
    hull = cv2.convexHull(contlist[0])
#            hullshape=hull.shape
#            hull[hullshape[0]][0][0]=hull[0][0][0]
#            hull[hullshape[0]][0][1]=hull[0][0][1]
    coordinates = zip(hull[:,0,1], hull[:,0,0])
    pts=list(coordinates)
    hullpts=copy.copy(pts)
    pts.append(pts[0])
    hullpts=copy.copy(pts)
    for pt in pts:
        sliceimg2[pt]=[0,255,0] 
    cv2.imwrite(pathhull+"hull"+fname, sliceimg2)

    
    
    #THE CONVEX HULL CAN BE DRAWN HERE
    longestdistancePointindices=calculateDistanceBetweenAllPairsandReturnLongestDistIndex(hull)
        
#        l=list1.index(find)
#        l=borderl1c.index(pts[longestdistancePointindices])
        
    for ii,jj in enumerate(borderl1c):
#                if((hull[longestdistancePointindices][0][0]==borderl1[ii][1]) & (hull[longestdistancePointindices][0][1]==borderl1[ii][0])):
        if((pts[longestdistancePointindices][1]==borderl1[ii][1]) & (pts[longestdistancePointindices][0]==borderl1[ii][0])):
            startindex=ii
            break
    for ii,jj in enumerate(borderl1c):
#                if((hull[longestdistancePointindices+1][0][0]==borderl1[ii][1]) & (hull[longestdistancePointindices+1][0][1]==borderl1[ii][0])):
        if((pts[longestdistancePointindices+1][1]==borderl1[ii][1]) &(pts[longestdistancePointindices+1][0]==borderl1[ii][0])):
            endindex=ii 
            break

    if(startindex>endindex):
#        del borderl1c1[startindex:endindex+1]
#        del borderl1c1[]
        temp=startindex
        startindex=endindex
        endindex=temp  

    lengthbord=len(borderl1c)
#CHECK IF THE RIGHT BORDER HAS BEEN DELETED    
    indices = [(startindex, endindex)]
    borderl1c1=[borderl1c[s:e+1] for s,e in indices] 
    borderl1c1=borderl1c1[0]   
    borderl1c2=copy.copy(borderl1c)
    borderl1c2=borderl1c2[endindex:lengthbord]+borderl1c2[0:startindex]
#    del borderl1c2[startindex:endindex+1]
    
    hullc=copy.copy(hull)
    del hullpts[longestdistancePointindices]
    del hullpts[longestdistancePointindices]
    
                
              
#    del borderl1c[startindex:endindex]
    
#    hullc=copy.copy(hull)
    
    intersectlist1=intersection(hullpts,borderl1c1)
    intersectlist2=intersection(hullpts,borderl1c2)
    if(len(intersectlist1)>len(intersectlist2)):
        borderl1c=borderl1c1
    else: 
        borderl1c=borderl1c2
        
#    if not intersectlist1:
#        borderl1c=borderl1c2
#    if  not intersectlist2:
#        borderl1c=borderl1c1
        
            
        
    
    imag=np.zeros((512,512,3))
    for pt in borderl1c:
        sliceimg[pt]=[0,0,255]    
#    for pt in border2:
#        imag[pt]=[0,255,0]   
#    plt.imsave("border"+fname,imag,cmap='gray')
    cv2.imwrite(pathnewborder+"newborder"+fname, sliceimg)

        
    return borderl1c
#        print("MATCH EXISTS")
#                if int(centroidl1[0])==j[0]:
#                    labposnl1.append(i)
#            cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)
#    # Display the final convex hull image
#    cv2.imshow('ConvexHull', img)
#    cv2.waitKey(0)                    
def getMaxMin(img):
    minimum=min(img.flatten())
    print(minimum)
    maximum=max(img.flatten())
    print(maximum) 
    return minimum,maximum
def png4(dcmimage):
#    ds = pydicom.dcmread(dcmimagename)
#    pa=ds.pixel_array
    ImgCopy=dcmimage
    minimum,maximum=getMaxMin(dcmimage)                 
    toadd=32767+minimum
    toadd=toadd.astype(np.int16)
    ImgCopy=ImgCopy+toadd
    ImgCopy=ImgCopy.astype(np.int16)
    pngImgCopy=np.floor(ImgCopy/256)
    return pngImgCopy
#    io.imsave(curref, pngImgCopy)    
#    cv2.imwrite(curref,pngImgCopy)
#    destination='/DATA/SG/Suji/implementation/journal2/png4_output/'
#    savefile(destination,dcmimagename,curref)    
    
def png2(dcmimage):
#    ds = pydicom.dcmread(dcmimagename)
    shape = dcmimage.shape
    # Convert to float to avoid overflow or underflow losses.
    image_2d = dcmimage.astype(float)
    # Rescaling grey scale between 0-255
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint
    image_2d_scaled = np.uint8(image_2d_scaled)
    # Write the PNG file
    return image_2d_scaled
#    with open(curref, 'wb') as png_file:
#        w = png.Writer(shape[1], shape[0], greyscale=True)
#        w.write(png_file, image_2d_scaled)
#    destination='/DATA/SG/Suji/implementation/journal2/png2_output/'
#    savefile(destination,dcmimagename,curref)    

def drawColContour(baseImg,gndIm):
    plt.figure(),plt.title('contour'),plt.imshow(baseImg, cmap=plt.cm.gray),#plt.show()
    cnimg=copy.copy(gndIm).astype(np.uint8)
    cnimg3c = cv2.cvtColor(cnimg, cv2.COLOR_GRAY2RGB)

    
    basimg=copy.copy(baseImg) 
    plt.figure(),plt.title('contour'),plt.imshow(basimg, cmap=plt.cm.gray)#,plt.show()
    contours, hierarchy = cv2.findContours(cnimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cntimg = cv2.drawContours(cnimg3c, contours, -1, (255,0,0), 2, cv2.LINE_8, hierarchy, 100);
    th_img=cntimg[:,:,0] 
    get_high_vals = th_img == 255
    basimg[get_high_vals] = (0, 255, 0)
    plt.figure(),plt.title('contour'),plt.imshow(basimg, cmap=plt.cm.gray)#,plt.show()
    return basimg    

def drawContour(baseImg,gndIm):
 #   plt.figure(),plt.title('contour'),plt.imshow(baseImg, cmap=plt.cm.gray),plt.show()
    cnimg=copy.copy(gndIm).astype(np.uint8)
    cnimg3c = cv2.cvtColor(cnimg, cv2.COLOR_GRAY2RGB)

    
    basimg=copy.copy(baseImg).astype(np.uint8)
    basimg3c=cv2.cvtColor(basimg, cv2.COLOR_GRAY2RGB)
   # plt.figure(),plt.title('contour'),plt.imshow(basimg3c, cmap=plt.cm.gray),plt.show()
    contours, hierarchy = cv2.findContours(cnimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cntimg = cv2.drawContours(cnimg3c, contours, -1, (255,0,0), 2, cv2.LINE_8, hierarchy, 100);
    
#    if(len(contours==0)):
 #   cntimg = cv.drawContours(cnimg3c, contours, -1, (255,0,0),2)#, lineType = cv.LINE_8)
    th_img=cntimg[:,:,0] 
    get_high_vals = th_img == 255
    basimg3c[get_high_vals] = (255, 0, 0)
   # plt.figure(),plt.title('contour'),plt.imshow(basimg3c, cmap=plt.cm.gray),plt.show()
    return basimg3c  

#def drawBordersAndSave(folder,bordert1,slicee,patid,fname):
#    imag=copy.copy(slicee)
#    imag=np.zeros((512,512,3),np.int8) 
##    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)
##    for pt in borderl2full:
##        imag[pt]=[0,0,255]
##    for pt in borderl1full:
##        imag[pt]=[0,255,0]   
##    cv2.imwrite(pathdistance+"l2fullRandl1fullB_"+fname, imag)    
#    for pt in bordert1:
#        imag[pt]=[255,0,0]
#    cv2.imwrite(folder+patid+fname,imag)
#    return imag
def removeEndPointUncertainities(borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21):
    E21c=copy.copy(E21)
    D21c=copy.copy(D21)
    E12c=copy.copy(E12)
    D12c=copy.copy(D12)
    
    borderl1c=copy.copy(borderl1)
    borderd12c=copy.copy(borderd12)
    
    borderl2c=copy.copy(borderl2)
    borderd21c=copy.copy(borderd21)
    m1 = np.mean(E12)
    a1 = np.average(E12)
    s1 = np.std(E12)
    v12= np.var(E12)
    min1=min(E12)
    max1=max(E12)
    thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21)
    a2 = np.average(E21)
    s2 = np.std(E21)
    v21= np.var(E21)
    min2=min(E21)
    max2=max(E21)
    bin_edges1=np.arange(0, 100, 1).tolist()
    thresh21 = threshold_otsu(E21) 
    if(E12[0]<max2):
        print("first point in 12 is ok")
    else:
        for index, itemvalue in enumerate(E12):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > max2):
                E12c=np.delete(E12c,1)
                D12c=np.delete(D12c,1)                
                borderl1c.pop(1)
                borderd12c.pop(1)
            else:
                break        
    if(E21[0]<max1):
        print("first point in 21 is ok")
    else:
        for index, itemvalue in enumerate(E21):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > max1):
                E21c=np.delete(E21c,1)
                D21c=np.delete(D21c,1)                
                borderl2c.pop(1)
                borderd21c.pop(1)
            else:
                break          
    if(E12[-1]<max2):
        print("last point in 12 is ok")
    else:
        for index, itemvalue in enumerate(reversed(E12)):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > max2):
                E12c=np.delete(E12c,-1)
                D12c=np.delete(D12c,-1)                
                borderl1c.pop(-1)
                borderd12c.pop(-1)
            else:
                break        
    if(E21[-1]<max1):
        print("last point in 21 is ok")
    else:
        for index, itemvalue in enumerate(reversed(E21)):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > max1):
                E21c=np.delete(E21c,-1)
                D21c=np.delete(D21c,-1)                
                borderl2c.pop(-1)
                borderd21c.pop(-1)
            else:
                break
        print(index)
        print(itemvalue)
        #update all the datastructures

            
        
    return borderl1,borderl2c,D12,E12,D21c,E21c,borderd12,borderd21c


#import matplotlib.pyplot as plt
def plotHistogram(probability,thresh,fname,patid):

    
#probability = [0.3602150537634409, 0.42028985507246375, 
#  0.373117033603708, 0.36813186813186816, 0.32517482517482516, 
#  0.4175257731958763, 0.41025641025641024, 0.39408866995073893, 
#  0.4143222506393862, 0.34, 0.391025641025641, 0.3130841121495327, 
#  0.35398230088495575]
    my_dpi=96
    plt.figure()
    plt.hist(probability, bins=probability.size-probability.size%100)
    plt.axvline(thresh, color='r')
    pathdistance='./outputnew/'+patid+'/a2p/histogram/'
#    pathdistance='./outputnew/'+patid+'/'
    plt.savefig(pathdistance+"Hist"+fname, dpi=my_dpi,transparent=True)
#    plt.show()


def plotRectifiedLungNodule1(borderl1,slice,fname,patid):
    pathdistance='./outputnew/'+patid+'/a2p/rectifiedlung/'
    my_dpi=96
    sliceimg= cv2.cvtColor(slice, cv2.COLOR_GRAY2BGR)
    print("5")
   # plt.figure(),plt.imshow(sliceimg)#,cmap=plt.cm.gray)
    global hull,longestdistancePointindices
    borderl1np=np.array(borderl1,dtype=np.int)
    hull = ConvexHull(borderl1np)
    
    imag=np.zeros((512,512)) 
    imagret=np.zeros((512,512)) 
    imag=slice
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in borderl1:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
    for i in hull.vertices:
        imagret[borderl1np[:,0][i],borderl1np[:,1][i]]=255
    from skimage.morphology import convex_hull_image

#        plt.scatter(borderl1np[:,0],borderl1np[:,1])
#        plt.plot(borderl1np[:,0][hull.vertices], borderl1np[:,1][hull.vertices])
    plt.plot(borderl1np[:,1][hull.vertices], borderl1np[:,0][hull.vertices])        
    plt.title("new border")
    plt.savefig(pathdistance+"TESTINGNEWBORDER"+fname, dpi=my_dpi,transparent=True)
    chull = convex_hull_image(imagret) 
    
    
    
#    chullcoord=np.nonzero(groupMatrix)
##    cchull=
#    longestdistancePointindices=calculateDistanceBetweenAllPairsandReturnLongestDistIndex(chullcoord)
    
    
    chull=chull*255
    chull=chull.astype(np.uint8)
    edges1, hierarchy = cv2.findContours(chull, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index,blockchain1=getBlockChainAsList(edges1) 
    blockchain1np=np.array(blockchain1,dtype=np.int)
    sha=blockchain1np.shape
    blockchain1tuple=blockchain1np.reshape(sha[0],1,sha[1])
    blockchain1np=np.asarray(blockchain1tuple)
    longestdistancePointindices=calculateDistanceBetweenAllPairsandReturnLongestDistIndex(blockchain1np)
    
    
    
    
    return blockchain1    
    
def plotRectifiedLungNodule(borderl1,slice,fname,patid):
    pathdistance='./outputnew/'+patid+'/a2p/rectifiedlung/'
    my_dpi=96
    borderl1np=np.array(borderl1,dtype=np.int)
    hull = ConvexHull(borderl1np)
    imag=np.zeros((512,512)) 
    imagret=np.zeros((512,512)) 
    imag=slice
    plt.figure()
    plt.imshow(imag, cmap='Greys')
    x=[]
    y=[]
    for i in borderl1:
        x.append(i[1])
        y.append(i[0])
    plt.plot(x,y) 
    for i in hull.vertices:
        imagret[borderl1np[:,0][i],borderl1np[:,1][i]]=255
    from skimage.morphology import convex_hull_image

#        plt.scatter(borderl1np[:,0],borderl1np[:,1])
#        plt.plot(borderl1np[:,0][hull.vertices], borderl1np[:,1][hull.vertices])
    plt.plot(borderl1np[:,1][hull.vertices], borderl1np[:,0][hull.vertices])        
    plt.title("new border")
    plt.savefig(pathdistance+"TESTINGNEWBORDER"+fname, dpi=my_dpi,transparent=True)
    chull = convex_hull_image(imagret) 
    
    
    
#    chullcoord=np.nonzero(groupMatrix)
##    cchull=
#    longestdistancePointindices=calculateDistanceBetweenAllPairsandReturnLongestDistIndex(chullcoord)
    
    
    chull=chull*255
    chull=chull.astype(np.uint8)
    edges1, hierarchy = cv2.findContours(chull, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index,blockchain1=getBlockChainAsList(edges1) 
    return blockchain1

#def getlobeborders(image):
#    plt.figure(),plt.imshow(image, cmap=plt.cm.gray)#, plt.show()
#    labelled1 = label(image)
#    regprop1=regionprops(labelled1)
#    alllabels = [r.label for r in regionprops(labelled1)]
#    areas = [r.area for r in regionprops(labelled1)]
#    centroids=[r.centroid for r in regionprops(labelled1)]
#    areas.sort()
#    firstmaxarea=areas[0]
#    secondmaxarea=areas[1]
#    lab1posn=[i for i, j in enumerate(areas) if j == firstmaxarea]
#    lab2posn=[i for i, j in enumerate(areas) if j == secondmaxarea]
#    lab1centroid=centroids[lab1posn[0]]
#    lab2centroid=centroids[lab2posn[0]]
#    lab1posn=lab1posn[0]+1
#    lab2posn=lab2posn[0]+1
#    
#    firstlobe=(labelled1==lab1posn)*255
#    secondlobe=(labelled1==lab2posn)*255
#    
#    firstlobe=firstlobe.astype(np.uint8)
#    secondlobe=secondlobe.astype(np.uint8)
#
#    edges1, hierarchy = cv2.findContours(firstlobe, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    index,blockchain1=getBlockChainAsList(edges1) 
#    
##    cv2.drawContours(imag, edges1, -1, (0, 255, 0), 1) 
#
#    edges2, hierarchy = cv2.findContours(secondlobe, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#    index,blockchain2=getBlockChainAsList(edges2) 
#    
#    if(lab1centroid[1]<lab2centroid[1]):
#        
#        return blockchain1,blockchain2,firstlobe,secondlobe
#    else:
#        return blockchain2,blockchain1,secondlobe,firstlobe   

def removePieces(borderl1,borderl2,fname,patid):
    pathdistance='./outputnew/'+patid+'/a2p/removepieces/'
    newlobel1=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in borderl1:
        newlobel1[pt]=255 
    newlobel2=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in borderl2:
        newlobel2[pt]=255         
    plt.figure(), plt.imshow(newlobel1, cmap=plt.cm.gray)#, plt.show()
    plt.figure(), plt.imshow(newlobel2, cmap=plt.cm.gray)#, plt.show()
    labelled1 = label(newlobel1)
    regprop1=regionprops(newlobel1)
    labelled2 = label(newlobel2)
    regprop2=regionprops(newlobel2)    
    
    alllabels1 = [r.label for r in regionprops(labelled1)]
    areas1 = [r.area for r in regionprops(labelled1)]
    centroids1=[r.centroid for r in regionprops(labelled1)]
    areas1.sort()
    alllabels2 = [r.label for r in regionprops(labelled2)]
    areas2 = [r.area for r in regionprops(labelled2)]
    centroids2=[r.centroid for r in regionprops(labelled2)]
    areas2.sort()
    
    maxarea1=areas1[0]
    maxarea2=areas2[0]

    lab1posn=[i for i, j in enumerate(areas1) if j == maxarea1]
    lab2posn=[i for i, j in enumerate(areas2) if j == maxarea2]
    
    
#    lab1centroid=centroids[lab1posn[0]]
#    lab2centroid=centroids[lab2posn[0]]
    lab1posn=lab1posn[0]+1
    lab2posn=lab2posn[0]+1
    
    lobe1=(labelled1==lab1posn)*255
    lobe2=(labelled2==lab2posn)*255
    
    lobe1=lobe1.astype(np.uint8)
    lobe2=lobe2.astype(np.uint8)
#
    edges1, hierarchy = cv2.findContours(lobe1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index1,blockchain1=getBlockChainAsList(edges1) 
    edges2, hierarchy = cv2.findContours(lobe2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index2,blockchain2=getBlockChainAsList(edges2)     
    newlobel1=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in blockchain1:
        newlobel1[pt]=255 
    newlobel2=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in blockchain2:
        newlobel2[pt]=255 
    cv2.imwrite(pathdistance+"L1:RemovePieces"+fname, newlobel1) 
    cv2.imwrite(pathdistance+"L2:RemovePieces"+fname, newlobel2) 
    return blockchain1,blockchain2    

def removeEndPointDifferences(borderl1,borderl2,slice1rgb,slice2rgb,patid,fname):
    loop=0
    oldborderl1=[]
    oldborderl2=[]
    newborderl1=borderl1
    newborderl2=borderl2
    while(len(oldborderl1)!=len(newborderl1) and len(oldborderl2)!=len(newborderl2)):
        loop=loop+1
        oldborderl1=borderl1
        oldborderl2=borderl2
        D12,E12,borderd12=pairwiseDistancesArgMinMin(borderl1,borderl2)
        D21,E21,borderd21=pairwiseDistancesArgMinMin(borderl2,borderl1)
        borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21=removeEndPointUncertainities3(slice1rgb,slice2rgb,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,str(loop)+"__"+fname)
      #  removePieces(borderl1,borderl2,patid,fname)
        newborderl1=borderl1
        newborderl2=borderl2
    return borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21 

def removeEndPointUncertainities3(slice1,slice2,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,fname):
    pathdistance='./outputnew/'+patid+'/a2p/endPointUncertainities/'
#    fname="BeforeRemovalL1"+fname
#    imag=drawBordersAndSave(pathdistance,borderl1,slice,patid,fname)
#    fname="BeforeRemovalL2"+fname
#    drawBordersAndSave(pathdistance,borderl2,imag,patid,fname)    
    imag=copy.copy(slice1)
    for pt in borderl1:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : b4_rem_Uncert_border"+fname, imag)
    imag=copy.copy(slice1)
    for pt in borderl2:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : b4_rem_Uncert_border"+fname, imag)    
    E21c=copy.copy(E21)
    D21c=copy.copy(D21)
    E12c=copy.copy(E12)
    D12c=copy.copy(D12)
    
    borderl1c=copy.copy(borderl1)
    borderd12c=copy.copy(borderd12)
    
    borderl2c=copy.copy(borderl2)
    borderd21c=copy.copy(borderd21)
    m1 = np.mean(E12c)
    a1 = np.average(E12c)
    s1 = np.std(E12c)
    v12= np.var(E12c)
    min1=min(E12c)
    max1=max(E12c)
 #   thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21c)
    a2 = np.average(E21c)
    s2 = np.std(E21c)
    v21= np.var(E21c)
    min2=min(E21c)
    max2=max(E21c)
    
    tempstartTF21=0
    tempstartTF12=0
    tempendTF21=0
    tempendTF12=0
    
    if(m1>=m2):
        m=m2
    else:
        m=m1
    if(s1>=s2):
        s=s2
    else:
        s=s1 
    th=m    
    TF12=E12c>th
    TF21=E21c>th
    lenTF12=len(TF12)
    lenTF21=len(TF21)    
    for index, itemvalue in enumerate(TF12):
        if(itemvalue==True):
            continue
        else:
            startTF12=index
            break
    for index, itemvalue in enumerate(TF21):
        if(itemvalue==True):
            continue
        else:
            startTF21=index
            break
    if(D12[startTF12]>=startTF21):
        tempstartTF21=D12[startTF12]
    if(D21[startTF21]>=startTF12):
        tempstartTF12=D21[startTF21]
    startTF21=int(tempstartTF21)
    startTF12=int(tempstartTF12)
    for index, itemvalue in enumerate(reversed(TF12)):
        if(itemvalue==True):
            continue
        else:
            tempendTF12=index
            endTF12=lenTF12-tempendTF12
            break

    for index, itemvalue in enumerate(reversed(TF21)):
        if(itemvalue==True):
            continue
        else:
            tempendTF21=index
            endTF21=lenTF21-tempendTF21
            break      
    if(D12[endTF12]<=endTF21):
        tempendTF21=D12[endTF12]
    if(D21[endTF21]<=endTF12):
        tempendTF12=D21[endTF21]
    endTF21=int(tempendTF21)
    endTF12=int(tempendTF12)
    
    borderl1c=borderl1c[startTF12:endTF12]
    borderl2c=borderl2c[startTF21:endTF21]
    E12c=E12c[startTF12:endTF12]
    E21c=E21c[startTF21:endTF21]
    D12c=D12c[startTF12:endTF12]
    D21c=D21c[startTF21:endTF21]
    borderd12c=borderd12c[startTF12:endTF12]
    borderd21c=borderd21c[startTF21:endTF21]
    TF12=E12c>th
    TF21=E21c>th    

    
    imag=copy.copy(slice1)    
    for pt in borderl1c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : After_rem_Uncert_border"+fname, imag)      
    imag=copy.copy(slice1)
    for pt in borderl2c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : After_rem_Uncert_border"+fname, imag)   
            

    return borderl1c,borderl2c,D12c,E12c,D21c,E21c,borderd12c,borderd21c
def removeEndPointUncertainities4(slice1,slice2,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,fname):
    pathdistance='./outputnew/'+patid+'/a2p/endPointUncertainities/'
#    fname="BeforeRemovalL1"+fname
#    imag=drawBordersAndSave(pathdistance,borderl1,slice,patid,fname)
#    fname="BeforeRemovalL2"+fname
#    drawBordersAndSave(pathdistance,borderl2,imag,patid,fname)    
    imag=copy.copy(slice1)
    for pt in borderl1:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : b4_rem_Uncert_border"+fname, imag)
    imag=copy.copy(slice1)
    for pt in borderl2:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : b4_rem_Uncert_border"+fname, imag)    
    E21c=copy.copy(E21)
    D21c=copy.copy(D21)
    E12c=copy.copy(E12)
    D12c=copy.copy(D12)
    
    borderl1c=copy.copy(borderl1)
    borderd12c=copy.copy(borderd12)
    
    borderl2c=copy.copy(borderl2)
    borderd21c=copy.copy(borderd21)
    m1 = np.mean(E12c)
    a1 = np.average(E12c)
    s1 = np.std(E12c)
    v12= np.var(E12c)
    min1=min(E12c)
    max1=max(E12c)
 #   thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21c)
    a2 = np.average(E21c)
    s2 = np.std(E21c)
    v21= np.var(E21c)
    min2=min(E21c)
    max2=max(E21c)
    
    tempstartTF21=0
    tempstartTF12=0
    tempendTF21=0
    tempendTF12=0
    
    if(m1>=m2):
        m=m2
    else:
        m=m1
    if(s1>=s2):
        s=s2
    else:
        s=s1 
    th=m    
    TF12=E12c>th
    TF21=E21c>th
    lenTF12=len(TF12)
    lenTF21=len(TF21)    
    for index, itemvalue in enumerate(TF12):
        if(itemvalue==True):
            continue
        else:
            startTF12=index
            break
    for index, itemvalue in enumerate(TF21):
        if(itemvalue==True):
            continue
        else:
            startTF21=index
            break
    print("startTF12  "+str(startTF12)+"startTF21"+str(startTF21))  
    print("startTF12  "+str(E12c[startTF12])+"startTF21"+str(E21c[startTF21]))
    print("startTF12  "+str(D12c[startTF12])+"startTF21"+str(D21c[startTF21]))    
#    spt2=D12[startTF12]
#    if(D21[spt2]==startTF12):
#        tempstartTF21=spt2
#        tempstartTF12=startTF12
#    
#    
#    
#    
#    if(D12[startTF12]>=startTF21):
#        tempstartTF21=D12[startTF12]
#    if(D21[startTF21]>=startTF12):
#        tempstartTF12=D21[startTF21]
#    startTF21=int(tempstartTF21)
#    startTF12=int(tempstartTF12)
    for index, itemvalue in enumerate(reversed(TF12)):
        if(itemvalue==True):
            continue
        else:
            tempendTF12=index
            endTF12=lenTF12-tempendTF12-1
            break

    for index, itemvalue in enumerate(reversed(TF21)):
        if(itemvalue==True):
            continue
        else:
            tempendTF21=index
            endTF21=lenTF21-tempendTF21-1
            break      
    print("endTF12  "+str(endTF12)+"endTF21  "+str(endTF21))
    print("endTF12  "+str(E12c[endTF12])+"endTF21  "+str(E21c[endTF21]))   
    print("endTF12  "+str(D12c[endTF12])+"endTF21  "+str(D21c[endTF21]))    
#    if(D12[endTF12]<=endTF21):
#        tempendTF21=D12[endTF12]
#    if(D21[endTF21]<=endTF12):
#        tempendTF12=D21[endTF21]
#    endTF21=int(tempendTF21)
#    endTF12=int(tempendTF12)
    
    borderl1c=borderl1c[startTF12:endTF12]
    borderl2c=borderl2c[startTF21:endTF21]
    E12c=E12c[startTF12:endTF12]
    E21c=E21c[startTF21:endTF21]
    D12c=D12c[startTF12:endTF12]
    D21c=D21c[startTF21:endTF21]
    borderd12c=borderd12c[startTF12:endTF12]
    borderd21c=borderd21c[startTF21:endTF21]
    TF12=E12c>th
    TF21=E21c>th    

    
    imag=copy.copy(slice1)    
    for pt in borderl1c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : After_rem_Uncert_border"+fname, imag)      
    imag=copy.copy(slice1)
    for pt in borderl2c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : After_rem_Uncert_border"+fname, imag)   
            

    return borderl1c,borderl2c,D12c,E12c,D21c,E21c,borderd12c,borderd21c
def removeEndPointUncertainities5(slice1,slice2,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,fname):
    pathdistance='./outputnew/'+patid+'/a2p/endPointUncertainities/'
#    fname="BeforeRemovalL1"+fname
#    imag=drawBordersAndSave(pathdistance,borderl1,slice,patid,fname)
#    fname="BeforeRemovalL2"+fname
#    drawBordersAndSave(pathdistance,borderl2,imag,patid,fname)    
    imag=copy.copy(slice1)
    for pt in borderl1:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : b4_rem_Uncert_border"+fname, imag)
    imag=copy.copy(slice1)
    for pt in borderl2:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : b4_rem_Uncert_border"+fname, imag)    
    E21c=copy.copy(E21)
    D21c=copy.copy(D21)
    E12c=copy.copy(E12)
    D12c=copy.copy(D12)
    
    borderl1c=copy.copy(borderl1)
    borderd12c=copy.copy(borderd12)
    
    borderl2c=copy.copy(borderl2)
    borderd21c=copy.copy(borderd21)
    m1 = np.mean(E12c)
    a1 = np.average(E12c)
    s1 = np.std(E12c)
    v12= np.var(E12c)
    min1=min(E12c)
    max1=max(E12c)
 #   thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21c)
    a2 = np.average(E21c)
    s2 = np.std(E21c)
    v21= np.var(E21c)
    min2=min(E21c)
    max2=max(E21c)
    
    tempstartTF21=0
    tempstartTF12=0
    tempendTF21=0
    tempendTF12=0
    
    if(m1>=m2):
        m=m2
    else:
        m=m1
    if(s1>=s2):
        s=s2
    else:
        s=s1 
    th=m    
    TF12=E12c>th
    TF21=E21c>th
    lenTF12=len(TF12)
    lenTF21=len(TF21)    
    for index, itemvalue in enumerate(TF12):
        if(itemvalue==True):
            continue
        else:
            startTF12=index
            break
    for index, itemvalue in enumerate(TF21):
        if(itemvalue==True):
            continue
        else:
            startTF21=index
            break
    
#    temstartTF12=startTF12
#    temstartTF21=startTF21
#    startTF12=D21c[startTF21]
#    startTF21=D12c[startTF12]
    
#    print("startTF12  "+str(startTF12)+"startTF21"+str(startTF21))  
#    print("startTF12  "+str(E12c[startTF12])+"startTF21"+str(E21c[startTF21]))
#    print("startTF12  "+str(D12c[startTF12])+"startTF21"+str(D21c[startTF21]))    
#    
##    spt2=D12[startTF12]
##    if(D21[spt2]==startTF12):
##        tempstartTF21=spt2
##        tempstartTF12=startTF12
##    
##    
##    
##    
##    if(D12[startTF12]>=startTF21):
##        tempstartTF21=D12[startTF12]
##    if(D21[startTF21]>=startTF12):
##        tempstartTF12=D21[startTF21]
##    startTF21=int(tempstartTF21)
##    startTF12=int(tempstartTF12)
    for index, itemvalue in enumerate(reversed(TF12)):
        if(itemvalue==True):
            continue
        else:
            tempendTF12=index
            endTF12=lenTF12-tempendTF12-1
            break

    for index, itemvalue in enumerate(reversed(TF21)):
        if(itemvalue==True):
            continue
        else:
            tempendTF21=index
            endTF21=lenTF21-tempendTF21-1
            break      

#    temendTF12=endTF12
#    temendTF21=endTF21
#    endTF12=D21c[endTF21]
#    endTF21=D12c[endTF12]

#    print("endTF12  "+str(endTF12)+"endTF21  "+str(endTF21))
#    print("endTF12  "+str(E12c[endTF12])+"endTF21  "+str(E21c[endTF21]))   
#    print("endTF12  "+str(D12c[endTF12])+"endTF21  "+str(D21c[endTF21]))    
##    if(D12[endTF12]<=endTF21):
##        tempendTF21=D12[endTF12]
##    if(D21[endTF21]<=endTF12):
##        tempendTF12=D21[endTF21]
##    endTF21=int(tempendTF21)
##    endTF12=int(tempendTF12)
    
    borderl1c=borderl1c[startTF12:endTF12]
    borderl2c=borderl2c[startTF21:endTF21]
    E12c=E12c[startTF12:endTF12]
    E21c=E21c[startTF21:endTF21]
    D12c=D12c[startTF12:endTF12]
    D21c=D21c[startTF21:endTF21]
    borderd12c=borderd12c[startTF12:endTF12]
    borderd21c=borderd21c[startTF21:endTF21]
    TF12=E12c>th
    TF21=E21c>th    

    
    imag=copy.copy(slice1)    
    for pt in borderl1c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : After_rem_Uncert_border"+fname, imag)      
    imag=copy.copy(slice1)
    for pt in borderl2c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : After_rem_Uncert_border"+fname, imag)   
            

    return borderl1c,borderl2c,D12c,E12c,D21c,E21c,borderd12c,borderd21c




def removeEndPointUncertainities2(slice1,slice2,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,fname):
    pathdistance='./outputnew/'+patid+'/a2p/endPointUncertainities/'
#    fname="BeforeRemovalL1"+fname
#    imag=drawBordersAndSave(pathdistance,borderl1,slice,patid,fname)
#    fname="BeforeRemovalL2"+fname
#    drawBordersAndSave(pathdistance,borderl2,imag,patid,fname)    
    imag=copy.copy(slice1)
    for pt in borderl1:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : b4_rem_Uncert_border"+fname, imag)
    imag=copy.copy(slice1)
    for pt in borderl2:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : b4_rem_Uncert_border"+fname, imag)    
    E21c=copy.copy(E21)
    D21c=copy.copy(D21)
    E12c=copy.copy(E12)
    D12c=copy.copy(D12)
    
    borderl1c=copy.copy(borderl1)
    borderd12c=copy.copy(borderd12)
    
    borderl2c=copy.copy(borderl2)
    borderd21c=copy.copy(borderd21)
    m1 = np.mean(E12)
    a1 = np.average(E12)
    s1 = np.std(E12)
    v12= np.var(E12)
    min1=min(E12)
    max1=max(E12)
    thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21)
    a2 = np.average(E21)
    s2 = np.std(E21)
    v21= np.var(E21)
    min2=min(E21)
    max2=max(E21)
    
    if(m1>=m2):
        m=m2
    else:
        m=m1
    if(s1>=s2):
        s=s2
    else:
        s=s1 
    th=m    
    TF12=E12>th
    TF21=E21>th
    lenTF12=len(TF12)
    lenTF21=len(TF21)    
    for index, itemvalue in enumerate(TF12):
        if(itemvalue==True):
            continue
        else:
            startTF12=index
            break
    for index, itemvalue in enumerate(TF21):
        if(itemvalue==True):
            continue
        else:
            startTF21=index
            break
#    if(E12[startTF12]>startTF21):
#        tempstartTF21=E12[startTF12]
#    if(E21[startTF21]>startTF12):
#        tempstartTF12=E21[startTF21]
#    startTF21=tempstartTF21
#    startTF12=tempstartTF12
    for index, itemvalue in enumerate(reversed(TF12)):
        if(itemvalue==True):
            continue
        else:
            tempendTF12=index
            endTF12=lenTF12-tempendTF12
            break

    for index, itemvalue in enumerate(reversed(TF21)):
        if(itemvalue==True):
            continue
        else:
            tempendTF21=index
            endTF21=lenTF21-tempendTF21
            break      
#    if(E12[endTF12]<endTF21):
#        tempendTF21=E12[endTF12]
#    if(E21[endTF21]>endTF12):
#        tempendTF12=E21[endTF21]
#    endTF21=tempendTF21
#    endTF12=tempendTF12
    
    borderl1c=borderl1c[startTF12:endTF12]
    borderl2c=borderl2c[startTF21:endTF21]
    E12c=E12c[startTF12:endTF12]
    E21c=E21c[startTF21:endTF21]
    D12c=D12c[startTF12:endTF12]
    D21c=D21c[startTF21:endTF21]
    borderd12c=borderd12c[startTF12:endTF12]
    borderd21c=borderd21c[startTF21:endTF21]
    TF12=E12c>th
    TF21=E21c>th    

    
    imag=copy.copy(slice1)    
    for pt in borderl1c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : After_rem_Uncert_border"+fname, imag)      
    imag=copy.copy(slice1)
    for pt in borderl2c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : After_rem_Uncert_border"+fname, imag)   
            

    return borderl1c,borderl2c,D12c,E12c,D21c,E21c,borderd12c,borderd21c

def removeEndPointUncertainities1(slice1,slice2,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,fname):
    pathdistance='./outputnew/'+patid+'/a2p/endPointUncertainities/'
#    fname="BeforeRemovalL1"+fname
#    imag=drawBordersAndSave(pathdistance,borderl1,slice,patid,fname)
#    fname="BeforeRemovalL2"+fname
#    drawBordersAndSave(pathdistance,borderl2,imag,patid,fname)    
    imag=copy.copy(slice1)
    for pt in borderl1:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : b4_rem_Uncert_border"+fname, imag)
    imag=copy.copy(slice1)
    for pt in borderl2:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : b4_rem_Uncert_border"+fname, imag)    
    E21c=copy.copy(E21)
    D21c=copy.copy(D21)
    E12c=copy.copy(E12)
    D12c=copy.copy(D12)
    
    borderl1c=copy.copy(borderl1)
    borderd12c=copy.copy(borderd12)
    
    borderl2c=copy.copy(borderl2)
    borderd21c=copy.copy(borderd21)
    m1 = np.mean(E12)
    a1 = np.average(E12)
    s1 = np.std(E12)
    v12= np.var(E12)
    min1=min(E12)
    max1=max(E12)
    thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21)
    a2 = np.average(E21)
    s2 = np.std(E21)
    v21= np.var(E21)
    min2=min(E21)
    max2=max(E21)
    
    TF12=E12>m1+s1
    TF21=E21>m2+s2
    
    print (str(min1)+"  "+str(max1)+"  "+str(min2)+"  "+str(max2))
    bin_edges1=np.arange(0, 100, 1).tolist()
    thresh21 = threshold_otsu(E21) 
    if(E12[0]<m1):
        print("first point in 12 is ok")
    else:
        for index, itemvalue in enumerate(E12):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > m1):
                print("point in 12 is NOT ok")
                E12c=np.delete(E12c,0)
                D12c=np.delete(D12c,0)                
                borderl1c.pop(0)
                borderd12c.pop(0)
            else:
                break        
    if(E21[0]<m2):
        print("first point in 21 is ok")
    else:
        for index, itemvalue in enumerate(E21):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > m2):
                print("point in 21 is NOT ok")
                E21c=np.delete(E21c,0)
                D21c=np.delete(D21c,0)                
                borderl2c.pop(0)
                borderd21c.pop(0)
            else:
                break          
    if(E12[-1]<m1):
        print("last point in 12 is ok")
    else:
        for index, itemvalue in enumerate(reversed(E12)):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > m1):
                print("point in 12 is NOT ok")
                E12c=np.delete(E12c,-1)
                D12c=np.delete(D12c,-1)                
                borderl1c.pop(-1)
                borderd12c.pop(-1)
            else:
                break        
    if(E21[-1]<m2):
        print("last point in 21 is ok")
    else:
        for index, itemvalue in enumerate(reversed(E21)):
#        for index,itemvalue in E21[::-1]:
            if(itemvalue > m2):
                print("point in 21 is NOT ok")
                E21c=np.delete(E21c,-1)
                D21c=np.delete(D21c,-1)                
                borderl2c.pop(-1)
                borderd21c.pop(-1)
            else:
                break
        print(index)
        print(itemvalue)
#    imag=copy.copy(slice1)
#    borderl1n=np.array(borderl1)
#    sha=borderl1n.shape
#    borderl1c=borderl1n.reshape(sha[0],1,sha[1])    
#    borderl1c= getBlockChainAsList(borderl1c) 
#
    TF12=E12>m1+s1
    TF21=E21>m2+s2        
    imag=copy.copy(slice1)    
    for pt in borderl1c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L1 : After_rem_Uncert_border"+fname, imag)      
    imag=copy.copy(slice1)
    for pt in borderl2c:
        imag[pt]=255    
    cv2.imwrite(pathdistance+"L2 : After_rem_Uncert_border"+fname, imag)      
#        update all the datastructures

#    fname="AfterRemovalL1"+fname
#    imag=drawBordersAndSave(pathdistance,borderl1,slice,patid,fname)
#    fname="AfterRemovalL2"+fname
#    drawBordersAndSave(pathdistance,borderl2c,imag,patid,fname)             
        
    return borderl1c,borderl2c,D12,E12,D21c,E21c,borderd12c,borderd21c

def removeEndPointDifferences(borderl1,borderl2,slice1rgb,slice2rgb,patid,fname):
    loop=0
    oldborderl1=[]
    oldborderl2=[]
    newborderl1=borderl1
    newborderl2=borderl2
    while(len(oldborderl1)!=len(newborderl1) and len(oldborderl2)!=len(newborderl2)):
        loop=loop+1
        oldborderl1=borderl1
        oldborderl2=borderl2
        D12,E12,borderd12=pairwiseDistancesArgMinMin(borderl1,borderl2)
        D21,E21,borderd21=pairwiseDistancesArgMinMin(borderl2,borderl1)
        borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21=removeEndPointUncertainities3(slice1rgb,slice2rgb,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,str(loop)+"__"+fname)
        removePieces(borderl1,borderl2,patid,fname)
        newborderl1=borderl1
        newborderl2=borderl2
    return borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21 

def removeEndPointDifferencesNoLoop(borderl1,borderl2,slice1rgb,slice2rgb,patid,fname):
    loop=0
    oldborderl1=[]
    oldborderl2=[]
#    newborderl1=borderl1
#    newborderl2=borderl2
#  #  while(len(oldborderl1)!=len(newborderl1) and len(oldborderl2)!=len(newborderl2)):
#    loop=loop+1
#    oldborderl1=borderl1
#    oldborderl2=borderl2
    D12,E12,borderd12=pairwiseDistancesArgMinMin(borderl1,borderl2)
    D21,E21,borderd21=pairwiseDistancesArgMinMin(borderl2,borderl1)
    borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21=removeEndPointUncertainities5(slice1rgb,slice2rgb,borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21,patid,str(loop)+"__"+fname)
 #   removePieces(borderl1,borderl2,patid,fname)
    D12,E12,borderd12=pairwiseDistancesArgMinMin(borderl1,borderl2)
    D21,E21,borderd21=pairwiseDistancesArgMinMin(borderl2,borderl1)    
#    newborderl1=borderl1
#    newborderl2=borderl2
    return borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21 


def LobeProcessing(borderl1,borderl2,borderl1full,borderl2full,lobel1,lobel2,slice1rgb,slice2rgb,fname,patid,sd):
    s1=s2=sd
    noduleTrF=False
    rectifiedborder=[]
    pathdistance='./outputnew/'+patid+'/'
#    slice1rgb=slice1
#    slice2rgb=slice2 
    
    exp_datal1=np.asarray(borderl1)
    num_datal2=np.asarray(borderl2)
#    D, E = pairwise_distances_argmin_min(exp_data, num_data, metric="euclidean")
#    D2 = pairwise_distances_argmin(exp_data, num_data, metric="euclidean")            
    D12,E12,borderd12=pairwiseDistancesArgMinMin(borderl1,borderl2) # in waning phase distance from 1 to 2 is to be used for drawing contour
    D21,E21,borderd21=pairwiseDistancesArgMinMin(borderl2,borderl1) # in waning phase distance from 2 to 1 is to be used for drawing nodule            
#    drawConsecBorders(borderl1,borderl2,slice1rgb,slice2rgb)

    m1 = np.mean(E12)
    a1 = np.average(E12)
    s1 = np.std(E12)
    v12= np.var(E12)
    min1=min(E12)
    max1=max(E12)
    thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21)
    a2 = np.average(E21)
    s2 = np.std(E21)
    v21= np.var(E21)
    min2=min(E21)
    max2=max(E21)
    bin_edges1=np.arange(0, 100, 1).tolist()
    thresh21 = threshold_otsu(E21)
    
    
#    plotHistogram(E12,thresh12,"12_"+fname,patid)
#    plotHistogram(E21,thresh21,"21_"+fname,patid)
    
#    borderl1,borderl2=createChain(borderl1,borderl2,fname,patid)

    drawConsecBordersOpenCV(borderl1,borderl2,fname,slice1rgb,patid)
    drawConsecBordersB2BOpenCV(borderl1,borderl2,borderd12,"b4_12_"+fname,slice1rgb,slice2rgb,patid)    
    drawConsecBordersB2BOpenCV(borderl2,borderl1,borderd21,"b4_21_"+fname,slice2rgb,slice1rgb,patid)
    
 #   borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21=removeEndPointDifferences(borderl1,borderl2,slice1rgb,slice2rgb,fname,patid)
#################
    borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21=removeEndPointDifferencesNoLoop(borderl1,borderl2,slice1rgb,slice2rgb,fname,patid)
################
    #borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21=removeEndPointUncertainities(borderl1,borderl2,D12,E12,D21,E21,borderd12,borderd21)
    #borderl2,borderl1,D21,E21,D12,E12,borderd21,borderd12=removeEndPointUncertainities1(slice2rgb,slice1rgb,borderl2,borderl1,D21,E21,D12,E12,borderd21,borderd12,patid,fname)
####################    
    drawConsecBordersOpenCV(borderl1,borderl2,fname,slice1rgb,patid)
    drawConsecBordersB2BOpenCV(borderl1,borderl2,borderd12,"after_12_"+fname,slice1rgb,slice2rgb,patid)    
    drawConsecBordersB2BOpenCV(borderl2,borderl1,borderd21,"after_21_"+fname,slice2rgb,slice1rgb,patid)

    D12,E12,borderd12=pairwiseDistancesArgMinMin(borderl1,borderl2) # in waning phase distance from 1 to 2 is to be used for drawing contour
    D21,E21,borderd21=pairwiseDistancesArgMinMin(borderl2,borderl1) # in waning phase distance from 2 to 1 is to be used for drawing nodule            
#############################
    m1 = np.mean(E12)
    a1 = np.average(E12)
    md1=np.median(E12)
    s1 = np.std(E12)
    v12= np.var(E12)
    min1=min(E12)
    max1=max(E12)
    thresh12 = threshold_otsu(E12)
    m2 = np.mean(E21)
    a2 = np.average(E21)
    md2=np.median(E21)
    s2 = np.std(E21)
    v21= np.var(E21)
    min2=min(E21)
    max2=max(E21)
#    print(m1,a1,s1,v12,min1,max1,thresh12)
#    print(m2,a2,s2,v21,min2,max2,thresh21)    
    print("STD DEV"+str(s1))
    print("STD DEV"+str(s2))
    bin_edges1=np.arange(0, 100, 1).tolist()
    thresh21 = threshold_otsu(E21)
    my_dpi=96
    E2tnf=E21>thresh21
    
# drawConsecBordersB2B(borderl2,borderl1,borderd21,"left221",slice1,slice2)
# For an end point of edge 1, find the nearest point in edge2, check if this 
# distance is higher than the maximum 
# of all distances from 2 to 1, then keep reducing the end point till 
# it becomes lower than the maximum of all distances from 2 to 1.
# If it is lower, then accept it.    

    
#    print(thresh21,v21)
    
# Draw the distance histogram
# Remove the edge which is longer, prune at the ends so that they become relatively nearby. (REMOVE END UNCERTAINTIES)
# Get some analytical results in terms of percentage of nodule overlap    
   # if ((s1< 2.0) and (s2 < 2.0)):
    if ((s1< 5.0) and (s2 < 5.0)):    
        print("Juxta-pleural Nodule Absent")
        rectifiedborder=borderl2full
        rectifiedlobel=lobel2
    else:
        print("*****************Juxta-pleural Nodule Present************************************"+fname) 
#        borderl1=AnteriorPosteriorDetection1(lobel1,rectifiedborder,slice1rgb,fname+"RECTA2P_L1.png",patid)
        noduleTrF=True
#        nodulenearestborder1,noduleborder2,image=drawConsecBordersB2BOnlyNodulesOpenCV(borderl2,borderl1,borderd21,slice1rgb,E2tnf,fname,patid)
        nodulenearestborder1,noduleborder2,imagetemp=drawConsecBordersB2BOnlyNodulesBlobOpenCV(borderl2,borderl1,borderd21,slice1rgb,E2tnf,fname,patid)
#        rectifiedborder=plotRectifiedLungNodule(borderl2,slice1rgb,fname,patid)  
        
#****************************************************************************** 
        score=calculatebestmatch(lobel1,lobel2,fname,patid)
        bordert1 =createnewcontour(lobel1,lobel2,score,patid,fname)
#            #drawAllThreeBoundaries(borderl1,borderl2,bordert1)
        lengthwidth=40
        boxes=extractROIbestMatch(lobel1,lobel2,score,lengthwidth,imagetemp,fname,patid)
        
#        bordert2=AnteriorPosteriorDetection1(lober2,bordert1,slice2rgb,fname+"A2P_Corrected_R2.png",patid)
        
#        nodulenearestborder1,noduleborder2,image=drawConsecBordersB2BOnlyNodulesOpenCV(borderl2,borderl1,borderd21,E2tnf,fname,patid)
#        
#        rectifiedborder=plotRectifiedLungNodule(borderl2,slice1rgb,fname,patid)  
#            
#            fname="_"+str(i-1)+"_"+str(i)+"_"
#            slice2=copy.copy(volcp[:,:,i])
#            slice2rgb=png2(slice2)
#            
#####################################
        bordert2,lobel2=borderCorrection(borderl1,borderl2,lobel1,lobel2,borderl1full,borderl2full,bordert1,boxes,slice2rgb,patid,fname+"BC.png",lengthwidth)
        
#        bordert2=borderl2

        
 ####################################     
   #     bordert2=AnteriorPosteriorDetection1(lobel2,bordert2,slice2rgb,fname+"A2P_Corrected_R2.png",patid)
        imag=np.zeros((512,512,3),dtype=np.uint8)
        for i in bordert2:
#        imag[i[0],i[1]]=[255,0,0]
            imag[i]=255
        cv2.imwrite(pathdistance+"t2newborder"+fname, imag) 
        rectifiedborder=bordert2 
        rectifiedlobel=lobel2          
#        bordert2=AnteriorPosteriorDetection1(lobet2,bordert2,slice2rgb,fname+"A2P_Corrected_R2.png",patid)
#            borderl1=bordert1
#            lobel1=lobel2
#******************************************************************************            
        
        
#        borderl1np=np.array(borderl1,dtype=np.int)
#        hull = ConvexHull(borderl1np)
#        imag=np.zeros((512,512))
#        
#        
##        plt.figure()
##        plt.imshow(imag, cmap='Greys')
##        x=[]
##        y=[]
##        for i in border1c:
##            x.append(i[1])
##            y.append(i[0])
##        plt.plot(x,y) 
##        x=[]
##        y=[]
##        plt.savefig(pathdistance+"TESTING"+fname, dpi=my_dpi,transparent=True)
##        #include code for multiple nodules
##        return borderlinec,border1c
#        
#        imag=slice1rgb
#        plt.figure()
#        plt.imshow(imag, cmap='Greys')
#    
#
#        x=[]
#        y=[]
#        for i in borderl1:
#            x.append(i[1])
#            y.append(i[0])
#        plt.plot(x,y) 
#        
##        plt.scatter(borderl1np[:,0],borderl1np[:,1])
##        plt.plot(borderl1np[:,0][hull.vertices], borderl1np[:,1][hull.vertices])
#        plt.plot(borderl1np[:,1][hull.vertices], borderl1np[:,0][hull.vertices])        
#        plt.title("new border")
#        plt.savefig(pathdistance+"TESTINGNEWBORDER"+fname, dpi=my_dpi,transparent=True)
        

#        E2tnf=E21>thresh21 #border 2 contains nodule # border 1 is free of nodule
#        #the following command draws the nodule border in border1  and select points on nearest to nodule in border 2 
#        nodulenearestborder1,noduleborder2=drawConsecBordersB2BOnlyNodules(borderl2,borderl1,borderd21,E2tnf,'filename',patid) 
#        #the nodules nearest to nodule border1 in border 2 may have duplicates, those duplicates are removed 
#        slimnodulenearestborder1=Remove(nodulenearestborder1)
#        #After removal of duplicates, still many intermediate points may be missing and missing
#        # points need to added to make a line. So try to find the first point and last point
#        # and try to find all intermediate points too. Now you get a connected line in border 1
#        # instead of some disconnected points
#        selectallpointsnodulenearestborder1=getPointsFromTo(borderl1,slimnodulenearestborder1)
#        # Draw the nodule border 1 and the border 2 line nearest to nodule border1
#        drawNoduleContourAndNoduleNearestBorderContour(selectallpointsnodulenearestborder1,noduleborder2,'filename',patid)
#        # Now create six more borders. by adding and subtracting 1,2,3.
#        drawAllnearestborders(selectallpointsnodulenearestborder1)
#        # Now check which border overlaps well with the nodule border and select that border.
##        noduletoucingborder1=checkBorder1TouchOverlapWithLobe2(selectallpointsnodulenearestborder1,lobel2)
#        print("hello")
#    cv2.imwrite(pathdistance+"_rectified_"+patid+fname, rectifiedborder)
    print(thresh21)
    bin_edges2=np.arange(0, 100, 1).tolist()
    return noduleTrF,rectifiedborder,rectifiedlobel,s1,s2
#    return noduleTrF,borderl1
def calculatebestmatch(lobe1,lobe2,fname,patid):
    pathdistance='./outputnew/'+patid+'/a2p/intermediate_best_matches/'
    cv2.imwrite(pathdistance+"_lobe1_"+patid+fname, lobe1)
    cv2.imwrite(pathdistance+"_lobe2_"+patid+fname, lobe2)
    lobe1 =  lobe1 > 128
    lobe2 =  lobe2 > 128
    rangevalue=[]
    score=[]
    for ii in range(-15,15):
        rangevalue.append(ii)
        selem = disk(np.abs(ii))              
        if(ii<1):
           eroded = erosion(lobe1, selem)
           eroded=eroded*255
           cv2.imwrite(pathdistance+"_TstFIT_"+fname+str(ii)+patid+".png", eroded)
           f1score=f1_score(lobe2,eroded,average='micro')
           print(f1score)
           score.append(f1score)
        if(ii>0):
#           eroded = erosion(lobe2, selem) 
           dilated = dilation(lobe1, selem)
           dilated=dilated*255
           cv2.imwrite(pathdistance+"_TstFIT_"+fname+str(ii)+patid+".png", dilated)
           f1score=f1_score(lobe2,dilated,average='micro')
           print(f1score)
           score.append(f1score) 
    maxpos = score.index(max(score))
    scorepos=rangevalue[maxpos]
    
    return scorepos
def drawShape(img, coordinates, color):
    # In order to draw our line in red
    img = skimage.color.gray2rgb(img)

    # Make sure the coordinates are expressed as integers
    coordinates = coordinates.astype(int)

    img[coordinates[:, 0], coordinates[:, 1]] = color

    return img
def extractROIbestMatch(lobe1,lobe2,score,lengthwidth,tempimage,fname,patid):
#    score=2
    pathdistance='./outputnew/'+patid+'/a2p/with_boxes/'
    print ("INTO Boxes")
    cv2.imwrite(pathdistance+"_lobe1_"+str(score)+patid+fname, lobe1)
    cv2.imwrite(pathdistance+"_lobe2_"+str(score)+patid+fname, lobe2)
    lobe1 =  lobe1 > 128
    lobe2 =  lobe2 > 128    
    if(score<0):
        selem = disk(np.abs(score))
        templobe=erosion(lobe1, selem)
        templobe=templobe*255
        cv2.imwrite(pathdistance+"BESTFIT_"+fname+str(score)+patid+".png", templobe)
    elif(score>=0):
        selem = disk(score)
        templobe= dilation(lobe1, selem)
        templobe=templobe*255
        cv2.imwrite(pathdistance+"BESTFIT_"+fname+str(score)+patid+".png", templobe)
    
#    plt.figure(), plt.imshow(lobe1, cmap='Greys'),plt.show() 
#    plt.figure(), plt.imshow(templobe, cmap='Greys'),plt.show() 
        
#    lobeldiff=lobe1-templobe
#    lobeldiff[lobeldiff<5] = 0
#       
#    opened = opening(lobeldiff, selem)
    tempimage = cv2.cvtColor(tempimage, cv2.COLOR_RGB2GRAY)    
    contours = skimage.measure.find_contours(tempimage, 0.5)
    

    bounding_boxes = []
    for contour in contours:
        Xmin = np.min(contour[:,0]-lengthwidth)
        Xmax = np.max(contour[:,0]+lengthwidth)
        Ymin = np.min(contour[:,1]-lengthwidth)
        Ymax = np.max(contour[:,1]+lengthwidth)
        bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])
        
    with_boxes  = np.copy(tempimage)
    for box in bounding_boxes:
        #[Xmin, Xmax, Ymin, Ymax]
        r = [box[0],box[1],box[1],box[0], box[0]]
        c = [box[3],box[3],box[2],box[2], box[3]]
        rr, cc = polygon_perimeter(r, c, with_boxes.shape)
        with_boxes[rr, cc] = 255 #set color white
    cv2.imwrite(pathdistance+"boxes_"+fname, with_boxes)
    print ("OUTOF Boxes")
#    plt.figure(), plt.imshow(opened, interpolation='nearest', cmap=plt.cm.gray)     
#    plt.figure(), plt.imshow(lobeldiff, interpolation='nearest', cmap=plt.cm.gray) 
#    plt.figure(), plt.imshow(with_boxes, interpolation='nearest', cmap=plt.cm.gray)
        
#    plt.figure(), plt.imshow(opened, cmap='Greys'),plt.show()     
#    plt.figure(), plt.imshow(lobeldiff, cmap='Greys'),plt.show()        
#    plt.figure(), plt.imshow(with_boxes, cmap='Greys'),plt.show()    
    
#    plt.figure(), plt.imshow(with_boxes, interpolation='nearest', cmap=plt.cm.gray)
    print("Hello")
    return bounding_boxes  
#    plt.show()

#    plt.figure(), plt.imshow(opened, cmap='Greys'),plt.show()              
#    import cv2 as cv
#    import random as rng
#    thresh = 100 # initial threshold
#    canny_output = cv.Canny(opened, thresh, thresh * 2)
#    contours,_ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#    contours_poly = [None]*len(contours)
#    boundRect = [None]*len(contours)
#    centers = [None]*len(contours)
#    radius = [None]*len(contours)
#    for i, c in enumerate(contours):
#        contours_poly[i] = cv.approxPolyDP(c, 3, True)
#        boundRect[i] = cv.boundingRect(contours_poly[i])
#        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly[i])
#    
#    
#    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
#    
#    
#    for i in range(len(contours)):
#        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#        cv.drawContours(drawing, contours_poly, i, color)
#        cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
#          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2
#    )
#        cv.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
#    
#    cv.imshow('Contours', drawing)    
    
def createnewcontour(lobel1,lobel2,score,patid,fname):
    
    pathdistance='./outputnew/'+patid+'/a2p/finalContour/'
    #    score=2
    if(score<0):
        selem = disk(np.abs(score))
        templobe=erosion(lobel1, selem)
    elif(score>=0):
        selem = disk(score)
        templobe= dilation(lobel1, selem)   
   
#    firstlobe=(labelled1==lab1posn)*255
#    secondlobe=(labelled1==lab2posn)*255
#    
#    firstlobe=firstlobe.astype(np.uint8)
#    secondlobe=secondlobe.astype(np.uint8)
#
    edges1, hierarchy = cv2.findContours(templobe, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index,blockchain1=getBlockChainAsList(edges1) 
    newlobe=np.zeros((512,512), dtype=np.uint8)
    newlobes=cv2.drawContours(newlobe,  edges1, contourIdx=-1, color=(255,255,255),thickness=-1)
 #   fname="test.png"
    cv2.imwrite(pathdistance+"boxes_"+fname, newlobes)
    return blockchain1
#    print("Hi")         
def intersection(lst1, lst2): 
    retlist=list(set(lst1) & set(lst2)) 
    return retlist
def getBoxIntIndex(borderlst,boxfirst,boxlast):
    lstindex1=borderlst.index(boxfirst)
    lstindex2=borderlst.index(boxlast)   
    if lstindex1>lstindex2:
        tempindex=lstindex2
        lstindex2=lstindex1
        lstindex1=tempindex
        
#        templst=retlist[1]
#        retlist[1]=retlist[0]
#        retlist[0]=templst
    print(lstindex1,lstindex2)
    return lstindex1,lstindex2

def restclipIntersecRegion(border,index1,index2):
    clipborder=border[index1:index2]
    restborder=border[0:index1]+border[index2:-1]
    return restborder, clipborder

class bres:
    def __init__ (self, p0, p1):
        self.initial = True
        self.end = False
        self.p0 = p0
        self.p1 = p1
        self.x0 = p0[0]
        self.y0 = p0[1]
        self.x1 = p1[0]
        self.y1 = p1[1]
        self.dx = abs(self.x1-self.x0)
        self.dy = abs(self.y1-self.y0)
        if self.x0 < self.x1:
            self.sx = 1
        else:
            self.sx = -1


        if self.y0 < self.y1:
            self.sy = 1
        else:
            self.sy = -1
        self.err = self.dx-self.dy

    def get_next (self):
        if self.initial:
            self.initial = False
            return [self.x0, self.y0]


        if self.x0 == self.x1 and self.y0 == self.y1:
            self.end = True
            return [self.x1, self.y1]
        self.e2 = 2*self.err
        if self.e2 > -self.dy:
            self.err = self.err - self.dy
            self.x0 = self.x0 + self.sx
        if self.e2 < self.dx:
            self.err = self.err + self.dx
            self.y0 = self.y0 + self.sy
        return [self.x0, self.y0]

    def get_current_pos (self):
        return [self.x0, self.y0]

    def finished (self):
        return self.end   

from skimage.draw import line as skimage_line
def bresenham_line(pstart, pstop):
    x0, y0 = pstart
    x0 = int(x0)
    y0 = int(y0)

    x1, y1 = pstop
    x1 = int(x1)
    y1 = int(y1)
    rr, cc = skimage_line(x0, y0, x1, y1)
    return [(r, c) for r, c in zip(rr, cc)]

def coloredBorders(borderl2,bordert1,indexl1,indexl2,indext1,indext2,patid,fname):
    pathdistance='./outputnew/'+patid+'/a2p/StringNewBorder/'
    newimglobel2=np.zeros((512,512,3), dtype=np.uint8)
    temp=borderl2[0:indexl1]
    for pt in temp:
        newimglobel2[pt]=[255,0,0]
    temp=borderl2[indexl1:indexl2]    
    for pt in temp:
        newimglobel2[pt]=[0,255,0]
    temp=borderl2[indexl2:] 
    for pt in temp:
        newimglobel2[pt]=[0,0,255]    
    cv2.imwrite(pathdistance+"TEST_L2_corrnlobe_"+fname, newimglobel2)
    newimglobet1=np.zeros((512,512,3), dtype=np.uint8)
    temp=bordert1[0:indext1]
    for pt in temp:
        newimglobet1[pt]=[255,0,0]
    temp=bordert1[indext1:indext2]    
    for pt in temp:
        newimglobet1[pt]=[0,255,0]
    temp=bordert1[indext2:] 
    for pt in temp:
        newimglobet1[pt]=[0,0,255]    
    cv2.imwrite(pathdistance+"TEST_T1_corrnlobe_"+fname, newimglobet1)    
def stringNewBorder(borderl2,bordert1,indexl1,indexl2,indext1,indext2,lengthwidth,patid,fname):
    print(indexl1)
    print(indexl2)
    print(indext1)
    print(indext2)
    coloredBorders(borderl2,bordert1,indexl1,indexl2,indext1,indext2,patid,fname)
    diffl=indexl2-indexl1
    difft=indext2-indext1
    pathdistance='./outputnew/'+patid+'/a2p/correct/'
    newlobe=np.zeros((512,512), dtype=np.uint8)
 #   newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    for pt in borderl2:
        newlobe[pt]=255          
    fname="test1.png"
    cv2.imwrite(pathdistance+"TESTcorrnlobe_"+fname, newlobe)
    print("hello") 

    newlobe=np.zeros((512,512), dtype=np.uint8)
    for pt in bordert1:
        newlobe[pt]=255          
    fname="test2.png"
    cv2.imwrite(pathdistance+"TESTcorrnlobe_"+fname, newlobe)
    print("hello") 

    
    if(difft > int(len(bordert1)/2)):
        clipborder=bordert1[indext2:]+bordert1[0:indext1]           
    else:
        clipborder=bordert1[indext1:indext2]   
   
    if(diffl > int(len(borderl2)/2)):
#        newborder=borderl2[indexl1:indexl2]+clipborder
        newborder=borderl2[indexl1:indexl2]
        print("End Points"+str(borderl2[indexl2])+"   "+str(clipborder[0]))
        w=bresenham_line(borderl2[indexl2], clipborder[0])
#        w = bres (borderl2[indexl2], clipborder[0])
#        templist=[]
#        while not w.finished ():
#            p = w.get_next()
#            pp=tuple(p)
#            templist.append(p)            
##            newborder=newborder+p#tuple(p)
#            print (p)    
        newborder=newborder+w 
        newborder=newborder+clipborder
        print("End Points"+str(clipborder[-1])+"   "+str(borderl2[indexl1]))
        w=bresenham_line(clipborder[-1], borderl2[indexl1])
#        w = bres (clipborder[-1], borderl2[indexl1])
#        templist=[]        
#        while not w.finished ():
#            p = w.get_next ()
#            pp=tuple(p)
#            templist.append(p)              
##            newborder=newborder+p#tuple(p)
#            print (p)  
        newborder=newborder+w            
    else:
#        newborder=borderl2[0:indexl1]+clipborder+borderl2[indexl2:]  
        newborder=borderl2[0:indexl1]
        print("End Points"+str(borderl2[indexl1])+"   "+str(clipborder[0]))
        w=bresenham_line(borderl2[indexl1-1], clipborder[0])
#        w = bres (borderl2[indexl1], clipborder[0])
#        templist=[]
#        for i in range(len(w)): 
##        while not w.finished ():
##            p = w.get_next ()
#            p = w[i]
#            pp=tuple(p)
#            templist.append(p)
##            templist.append(tuple([p(0),p[1]]))
##            newborder=newborder+p#tuple(p) 
#            print (p)    
##        newborder=newborder+templist    
        newborder=newborder+w                
        newborder=newborder+clipborder 
        print("End Points"+str(clipborder[-1])+"   "+str(borderl2[indexl2]))
        w=bresenham_line(clipborder[-1], borderl2[indexl2])
#        w = bres (clipborder[-1], borderl2[indexl2])
#        templist=[]  
#        for i in range(len(w)):        
##        while not w.finished ():
##            p = w.get_next ()
#            p = w[i]
#            pp=tuple(p)
#            templist.append(p)             
##            newborder=newborder+p#tuple(p) 
#            print (p)   
          
        newborder=newborder+w  
        newborder=newborder+borderl2[indexl2:] 
        print()

    newlobe=np.zeros((512,512), dtype=np.uint8)
    for pt in newborder:
        newlobe[pt]=255  
    for pt in clipborder:
        newlobe[pt]=255          
    fname="test.png"
    cv2.imwrite(pathdistance+"TESTcorrnlobe_"+fname, newlobe)
    contours, hierarchy = cv2.findContours(newlobe,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #index,blockchain2=getBlockChainAsList(edges2) 

    newlobes=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    
    h, w = newlobe.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(newlobes, mask, (0,0), 127);
    cv2.imwrite(pathdistance+"TESTcorrnlobe_FILLED_"+fname, newlobes)
    print("hello") 
    
    trufal=newlobes==0
    newlobes[trufal]=255
    trufal=newlobes==127
    newlobes[trufal]=0    
    edges1, hierarchy = cv2.findContours(newlobes, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    index,newnewborder=getBlockChainAsList(edges1) 
           
#    nbfirstend=borderl2[indexl1]
#    nbsecondend=borderl2[indexl2]
#    cbfirstend=clipborder[0]
#    cbsecondend=clipborder[1]
    
  
    
    


    
#    if(diffl < int(len(borderl2)/2)):
#        newborder=borderl2[indexl1:indexl2]+clipborder
#    else:
#        newborder=borderl2[0:indexl1]+clipborder+borderl2[indexl2:-1] 
        
#    intersect=intersection(clipborder,newborder)
#    newlobe=np.zeros((512,512), dtype=np.uint8)
#    for pt in newborder:
#        newlobe[pt]=255  
#    contours, hierarchy = cv2.findContours(newlobe,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
#    intert=intersection(bordert1, boxlist)
        
    
#    newborder=borderl2[0:indexl1]+bordert1[indext1:indext2]+borderl2[indexl2:-1]
#    newborder=bordert1[indext1:indext2]
    return clipborder,newborder,newlobes    
    #return clipborder      
def getFarthestPoints(pts):
    cords=copy.copy(pts)
    length=len(cords) 
    maxdist=0.0
    oneend=pts[0]
    lastend=pts[0]
    for i in range(length): 
        pto=cords[i]
        for j in range(length): 
            pte=pts[j]
            dst = euclidean(pto,pte)
            if(dst>maxdist):
                maxdist=dst
                oneend=pto
                lastend=pte
    return oneend,lastend,maxdist
                
    
    
##    cords.append(cords[0])
#    dist=[]   
#    for i in range(length+1): 
#        if (i==0):
#            beg=cords[i]
#            continue
#        else:
#            nxt=cords[i]
#            dst = euclidean(beg,nxt)
#            beg=nxt
#            dist.append(dst)
#    maxpos = dist.index(max(dist)) 
#    
#    if(maxpos+1==length):
#        last=0
#    else:
#        last=maxpos+1 
#    return maxpos,last,dist    
def borderCorrection(borderl1,borderl2,lobel1,lobel2,borderl1full,borderl2full,bordert1,bounding_boxes,slice2,patid,fname,lengthwidth):
    pathdistance='./outputnew/'+patid+'/a2p/correct/'
    newimgborder=np.zeros((512,512), dtype=np.uint8)
    my_dpi=96
    imag=copy.copy(slice2)
    imag = cv2.cvtColor(imag, cv2.COLOR_GRAY2RGB)
#    for pt in borderl2full:
#        imag[pt]=[0,0,255]
#    for pt in borderl1full:
#        imag[pt]=[0,255,0]   
#    cv2.imwrite(pathdistance+"l2fullRandl1fullB_"+fname, imag)    
    for pt in bordert1:
        imag[pt]=[255,0,0]
    for pt in borderl2full:
        imag[pt]=[0,255,0]
    cv2.imwrite(pathdistance+"t1Bandl2G_"+fname, imag) 
    pathboth='./outputnew/'+patid+'/a2p/corrected/'    
    itera=0
    for box in bounding_boxes:
        itera=itera+1
        boxlist=[]
        #[Xmin, Xmax, Ymin, Ymax]
        print("godfrey")
        r = [box[0],box[1],box[1],box[0], box[0]]
        c = [box[3],box[3],box[2],box[2], box[3]]
        rr, cc = polygon_perimeter(r, c, imag.shape)
        imag[rr, cc] = [255,255,255] #set color white  
        cv2.imwrite(pathdistance+"b4corrn_"+fname, imag)   
        a=int(box[0])
        b=int(box[1])
        c=int(box[2])
        d=int(box[3])
        for i in range (c,d):
            boxlist.append((a,i))
        for i in range (c,d):
            boxlist.append((b,i))      
        for i in range (a,b):
            boxlist.append((i,c))
        for i in range (a,b):
            boxlist.append((i,d))               
#        print(len(boxlist)) 
#        print("GOOD")
        interl=intersection(borderl2full, boxlist)
        intert=intersection(bordert1, boxlist)   
        if(not interl or not intert):
            return borderl2,lobel2
        inter=[]
        if(len(interl) >2):
            first,last,dist=getFarthestPoints(interl)
            inter.append(first)
            inter.append(last)
            interl=inter
        inter=[]
        if(len(intert) >2):
            first,last,dist=getFarthestPoints(intert)
            inter.append(first)
            inter.append(last)     
            intert=inter
#        inter=[]
#        if(len(interl) >2):
#            inter.append(interl[0])
#            inter.append(interl[-1])
#        if(len(interl) >2):
#            interl=inter    
#        inter=[]
#        if(len(intert) >2):
#            inter.append(intert[0])
#            inter.append(intert[-1])   
#        if(len(intert) >2):
#            intert=inter         
#        print(len(interl))    
#        print(len(intert))   
#        if(len(interl)==2 and len(intert)==2):
        indexl1,indexl2=getBoxIntIndex(borderl2full,interl[0],interl[1])
        indext1,indext2=getBoxIntIndex(bordert1,intert[0],intert[1])
#        restborderl2,clipborderl2=restclipIntersecRegion(borderl2,indexl1,indexl2)
#        restbordert1,clipbordert1=restclipIntersecRegion(bordert1,indext1,indext2)
        clipborder,newborder,lobe=stringNewBorder(borderl2full,bordert1,indexl1,indexl2,indext1,indext2,lengthwidth,patid,fname)
        for pt in clipborder:
            imag[pt]=[255,0,0]
        
        for pt in newborder:
            newimgborder[pt]=255
#        lobe=~lobe    
#        cv2.imwrite(pathdistance+"lobe_"+fname, img_pl)            
        
#        contours,hierarchy = cv2.findContours(newlobe,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#        newlobe=cv2.drawContours(newlobe, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
        
#        contour = np.array(newborder) 
#        contours=contour.reshape((contour.shape[0],1,contour.shape[1]))
#        img_pl = np.zeros((255,255))
#        cv2.fillPoly(img_pl,pts=contours,color=(255))    
#        cv2.imwrite(pathdistance+"aftercorrntestlobe_"+fname, img_pl)
#        clipbordert1=clipIntersecRegion(bordert1,intert)
#        borderl2=joinbordertONborderl(restborderl2,clipbordert1)

#        areas = [r.area for r in regionprops(imag)]
        
        
#        print(intert)
#        print(interl)
        print("hello")   
    #    _, contours, _ = cv2.findContours(cir.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#        contour = np.array(newborder) 
#        contour=contour.reshape((contour.shape[0],1,contour.shape[1]))
#        cv2.drawContours(imag, contour, contourIdx=-1, color=(255,255,255),thickness=-1)

#        h, w = imag.shape[:2]
#        mask = np.zeros((h+2, w+2), np.uint8)
#        cv2.floodFill(newlobe, mask, (0,0), 0);
        cv2.imwrite(pathdistance+"aftercorrnlobe_"+fname, lobe)
        cv2.imwrite(pathdistance+"aftercorrnborder_"+fname, newimgborder)
        cv2.imwrite(pathdistance+"aftercorrn_"+fname, imag)       
    return newborder,lobe
#    xl2=[]
#    yl2=[]
#    for i in borderl2:
#        if(box[0]<=i[1]<=box[1] and box[3]<=i[1]<=box[2]):
#        xl2.append(i[1])
#        yl2.append(i[0]) 
#    xt1=[]
#    yt1=[]        
#    for i in bordert1:
#        if(box[0]<=i[1]<=box[1] and box[3]<=i[1]<=box[2]):
#        xt1.append(i[1])
#        yt1.append(i[0]) 

             
def main():
    global nsmaskvolume,names,resultsubstring, df,printstr,segrefImgCopy,segcurImgCopy,GroundtruthVolume,SegmentedImageVolume,norsrccurImg,norsrcrefImg,norsegcurImg,norsegrefImg,scan,dcm,dcmintercept,dcmslope,vol,nods,refImgCopy,curImgCopy,grndImgCopy,referenceImagePosn,currentImagePosn
    scan = pl.query(pl.Scan)
    names= ['Patid','prevslice','nextslice','JPLYrN','JPRYrN','12LY','21LY','12RY','21RY','tnl','fpl','fnl','tpl','tnr','fpr','fnr','tpr','tnll','fpll','fnll','tpll','tnrr','fprr','fnrr','tprr','pllINTgtllln','pllUNIgtllln','pllDIFFgtllln','gtllDIFFpllln','vofl','gtlll','osrl','usrl','pllINTgtllrn','pllUNIgtllrn','pllDIFFgtllrn','gtllDIFFpllrn','vofr','gtllr','osrr','usrr']

                
    df = pd.DataFrame([], columns=names)
    lst = []
    
    lisst=['LIDC-IDRI-0007','LIDC-IDRI-0015','LIDC-IDRI-0018','LIDC-IDRI-0019','LIDC-IDRI-0020','LIDC-IDRI-0021','LIDC-IDRI-0022','LIDC-IDRI-0023','LIDC-IDRI-0027','LIDC-IDRI-0029','LIDC-IDRI-0033','LIDC-IDRI-0037','LIDC-IDRI-0041','LIDC-IDRI-0043','LIDC-IDRI-0045','LIDC-IDRI-0046','LIDC-IDRI-0053','LIDC-IDRI-0055','LIDC-IDRI-0057','LIDC-IDRI-0058','LIDC-IDRI-0060','LIDC-IDRI-0061','LIDC-IDRI-0061','LIDC-IDRI-0061','LIDC-IDRI-0061','LIDC-IDRI-0061','LIDC-IDRI-0067','LIDC-IDRI-0068','LIDC-IDRI-0068','LIDC-IDRI-0068','LIDC-IDRI-0073','LIDC-IDRI-0073','LIDC-IDRI-0074','LIDC-IDRI-0077','LIDC-IDRI-0078','LIDC-IDRI-0079','LIDC-IDRI-0080','LIDC-IDRI-0080','LIDC-IDRI-0081','LIDC-IDRI-0082','LIDC-IDRI-0087','LIDC-IDRI-0091','LIDC-IDRI-0095','LIDC-IDRI-0109','LIDC-IDRI-0115','LIDC-IDRI-0118','LIDC-IDRI-0120','LIDC-IDRI-0120','LIDC-IDRI-0120','LIDC-IDRI-0120','LIDC-IDRI-0121','LIDC-IDRI-0127','LIDC-IDRI-0128','LIDC-IDRI-0129','LIDC-IDRI-0133','LIDC-IDRI-0133','LIDC-IDRI-0134','LIDC-IDRI-0136','LIDC-IDRI-0137','LIDC-IDRI-0138','LIDC-IDRI-0139','LIDC-IDRI-0139','LIDC-IDRI-0141','LIDC-IDRI-0144','LIDC-IDRI-0147','LIDC-IDRI-0147','LIDC-IDRI-0150','LIDC-IDRI-0154','LIDC-IDRI-0154','LIDC-IDRI-0158','LIDC-IDRI-0158','LIDC-IDRI-0160','LIDC-IDRI-0163','LIDC-IDRI-0166','LIDC-IDRI-0170','LIDC-IDRI-0186','LIDC-IDRI-0186','LIDC-IDRI-0187','LIDC-IDRI-0188','LIDC-IDRI-0190','LIDC-IDRI-0190','LIDC-IDRI-0191','LIDC-IDRI-0192','LIDC-IDRI-0195','LIDC-IDRI-0195','LIDC-IDRI-0195','LIDC-IDRI-0198','LIDC-IDRI-0201','LIDC-IDRI-0207','LIDC-IDRI-0213','LIDC-IDRI-0220','LIDC-IDRI-0223','LIDC-IDRI-0223','LIDC-IDRI-0229','LIDC-IDRI-0229','LIDC-IDRI-0232','LIDC-IDRI-0235','LIDC-IDRI-0243','LIDC-IDRI-0244','LIDC-IDRI-0249','LIDC-IDRI-0257','LIDC-IDRI-0259','LIDC-IDRI-0260','LIDC-IDRI-0265','LIDC-IDRI-0267','LIDC-IDRI-0273','LIDC-IDRI-0285','LIDC-IDRI-0286','LIDC-IDRI-0289','LIDC-IDRI-0291','LIDC-IDRI-0297','LIDC-IDRI-0298','LIDC-IDRI-0303','LIDC-IDRI-0309','LIDC-IDRI-0310','LIDC-IDRI-0311','LIDC-IDRI-0312','LIDC-IDRI-0312','LIDC-IDRI-0315','LIDC-IDRI-0315','LIDC-IDRI-0319','LIDC-IDRI-0321','LIDC-IDRI-0323','LIDC-IDRI-0324','LIDC-IDRI-0325','LIDC-IDRI-0334','LIDC-IDRI-0340','LIDC-IDRI-0344','LIDC-IDRI-0355','LIDC-IDRI-0355','LIDC-IDRI-0356','LIDC-IDRI-0359','LIDC-IDRI-0362','LIDC-IDRI-0368','LIDC-IDRI-0368','LIDC-IDRI-0368','LIDC-IDRI-0368','LIDC-IDRI-0377','LIDC-IDRI-0379','LIDC-IDRI-0384','LIDC-IDRI-0385','LIDC-IDRI-0403','LIDC-IDRI-0404','LIDC-IDRI-0404','LIDC-IDRI-0404','LIDC-IDRI-0405','LIDC-IDRI-0407','LIDC-IDRI-0409','LIDC-IDRI-0409','LIDC-IDRI-0414','LIDC-IDRI-0415','LIDC-IDRI-0415','LIDC-IDRI-0415','LIDC-IDRI-0415','LIDC-IDRI-0421','LIDC-IDRI-0432','LIDC-IDRI-0437','LIDC-IDRI-0437','LIDC-IDRI-0447','LIDC-IDRI-0450','LIDC-IDRI-0450','LIDC-IDRI-0459','LIDC-IDRI-0462','LIDC-IDRI-0463','LIDC-IDRI-0466','LIDC-IDRI-0466','LIDC-IDRI-0469','LIDC-IDRI-0471','LIDC-IDRI-0471','LIDC-IDRI-0474','LIDC-IDRI-0475','LIDC-IDRI-0476','LIDC-IDRI-0477','LIDC-IDRI-0477','LIDC-IDRI-0481','LIDC-IDRI-0481','LIDC-IDRI-0481','LIDC-IDRI-0484','LIDC-IDRI-0484','LIDC-IDRI-0484','LIDC-IDRI-0484','LIDC-IDRI-0487','LIDC-IDRI-0487','LIDC-IDRI-0488','LIDC-IDRI-0489','LIDC-IDRI-0489','LIDC-IDRI-0489','LIDC-IDRI-0493','LIDC-IDRI-0494','LIDC-IDRI-0496','LIDC-IDRI-0504','LIDC-IDRI-0508','LIDC-IDRI-0509','LIDC-IDRI-0510','LIDC-IDRI-0510','LIDC-IDRI-0510','LIDC-IDRI-0523','LIDC-IDRI-0523','LIDC-IDRI-0523','LIDC-IDRI-0523','LIDC-IDRI-0526','LIDC-IDRI-0527','LIDC-IDRI-0529','LIDC-IDRI-0529','LIDC-IDRI-0534','LIDC-IDRI-0543','LIDC-IDRI-0543','LIDC-IDRI-0545','LIDC-IDRI-0550','LIDC-IDRI-0571','LIDC-IDRI-0575','LIDC-IDRI-0576','LIDC-IDRI-0583','LIDC-IDRI-0583','LIDC-IDRI-0583','LIDC-IDRI-0594','LIDC-IDRI-0594','LIDC-IDRI-0594','LIDC-IDRI-0597','LIDC-IDRI-0601','LIDC-IDRI-0601','LIDC-IDRI-0601','LIDC-IDRI-0604','LIDC-IDRI-0605','LIDC-IDRI-0608','LIDC-IDRI-0610','LIDC-IDRI-0613','LIDC-IDRI-0617','LIDC-IDRI-0619','LIDC-IDRI-0624','LIDC-IDRI-0624','LIDC-IDRI-0629','LIDC-IDRI-0634','LIDC-IDRI-0636','LIDC-IDRI-0641','LIDC-IDRI-0642','LIDC-IDRI-0643','LIDC-IDRI-0645','LIDC-IDRI-0650','LIDC-IDRI-0651','LIDC-IDRI-0660','LIDC-IDRI-0661','LIDC-IDRI-0688','LIDC-IDRI-0695','LIDC-IDRI-0697','LIDC-IDRI-0701','LIDC-IDRI-0702','LIDC-IDRI-0703','LIDC-IDRI-0705','LIDC-IDRI-0705','LIDC-IDRI-0709','LIDC-IDRI-0709','LIDC-IDRI-0713','LIDC-IDRI-0714','LIDC-IDRI-0719','LIDC-IDRI-0724','LIDC-IDRI-0733','LIDC-IDRI-0741','LIDC-IDRI-0741','LIDC-IDRI-0741','LIDC-IDRI-0742','LIDC-IDRI-0742','LIDC-IDRI-0742','LIDC-IDRI-0748','LIDC-IDRI-0749','LIDC-IDRI-0750','LIDC-IDRI-0751','LIDC-IDRI-0751','LIDC-IDRI-0751','LIDC-IDRI-0761','LIDC-IDRI-0765','LIDC-IDRI-0770','LIDC-IDRI-0773','LIDC-IDRI-0775','LIDC-IDRI-0776','LIDC-IDRI-0785','LIDC-IDRI-0790','LIDC-IDRI-0790','LIDC-IDRI-0798','LIDC-IDRI-0801','LIDC-IDRI-0801','LIDC-IDRI-0809','LIDC-IDRI-0811','LIDC-IDRI-0814','LIDC-IDRI-0819','LIDC-IDRI-0821','LIDC-IDRI-0834','LIDC-IDRI-0838','LIDC-IDRI-0843','LIDC-IDRI-0849','LIDC-IDRI-0849','LIDC-IDRI-0850','LIDC-IDRI-0858','LIDC-IDRI-0861','LIDC-IDRI-0864','LIDC-IDRI-0866','LIDC-IDRI-0871','LIDC-IDRI-0880','LIDC-IDRI-0882','LIDC-IDRI-0883','LIDC-IDRI-0886','LIDC-IDRI-0890','LIDC-IDRI-0893','LIDC-IDRI-0894','LIDC-IDRI-0896','LIDC-IDRI-0902','LIDC-IDRI-0905','LIDC-IDRI-0906','LIDC-IDRI-0912','LIDC-IDRI-0913','LIDC-IDRI-0921','LIDC-IDRI-0921','LIDC-IDRI-0923','LIDC-IDRI-0924','LIDC-IDRI-0925','LIDC-IDRI-0939','LIDC-IDRI-0942','LIDC-IDRI-0949','LIDC-IDRI-0951','LIDC-IDRI-0962','LIDC-IDRI-0971','LIDC-IDRI-0973','LIDC-IDRI-0973','LIDC-IDRI-0976','LIDC-IDRI-0978','LIDC-IDRI-0978','LIDC-IDRI-0978','LIDC-IDRI-0982','LIDC-IDRI-0985','LIDC-IDRI-0998','LIDC-IDRI-0998','LIDC-IDRI-0998','LIDC-IDRI-0998','LIDC-IDRI-1003','LIDC-IDRI-1004','LIDC-IDRI-1009']
    nodno=['1','0','3','0','0','1','0','0','5','0','1','0','0','1','1','2','0','6','0','2','5','0','1','3','4','5','0','2','4','5','0','2','0','0','3','0','0','1','1','0','0','1','0','0','0','0','0','1','2','3','0','0','2','10','0','1','1','6','1','0','0','1','0','0','1','6','1','0','1','0','3','2','0','0','1','3','4','3','3','0','1','0','2','0','4','5','0','2','0','0','2','0','2','2','6','0','1','0','2','0','0','0','1','0','0','0','1','0','1','0','0','1','0','2','0','0','0','1','5','5','0','0','0','1','0','1','0','1','0','0','3','0','1','0','3','4','5','2','0','2','1','0','3','4','5','1','1','1','3','0','1','2','4','7','0','0','0','1','1','4','6','0','2','0','0','1','0','0','1','0','0','1','0','1','4','5','6','0','1','0','1','0','6','1','0','2','6','1','0','0','0','0','4','0','1','2','0','1','3','4','5','0','0','2','0','0','1','1','0','0','0','0','1','16','19','1','2','3','0','1','2','3','0','4','0','1','0','1','1','4','5','0','0','0','6','0','2','0','0','2','0','2','2','0','0','3','1','0','0','1','2','3','6','0','0','1','5','0','1','5','3','4','13','1','2','2','0','2','6','3','0','0','5','5','0','7','0','1','2','0','3','0','0','0','2','1','0','1','3','5','6','0','10','1','1','1','5','2','0','2','2','2','0','1','0','0','0','0','6','2','3','4','0','1','0','2','1','0','0','1','0','0','1','1','0','2','3','2','0','0','1','4','5','6','2','0']
    startsliceno=['118','168','107','253','154','60','100','112','111','169','107','79','45','82','40','36','32','122','153','103','90','36','47','87','89','85','110','153','203','210','55','107','77','67','64','67','76','90','93','166','103','56','186','108','88','65','61','97','95','147','155','68','170','116','35','130','65','122','98','172','24','27','148','22','30','50','88','36','40','50','139','63','35','49','107','111','117','118','54','45','88','84','115','37','81','83','29','57','65','89','154','39','191','100','108','53','74','55','82','27','66','29','55','99','32','47','100','41','51','80','185','215','52','201','20','143','47','51','83','83','38','130','50','191','134','53','31','291','46','46','73','27','145','31','64','67','86','182','46','444','345','142','128','133','166','105','92','48','134','64','64','77','88','145','36','444','157','203','64','154','163','43','330','135','65','65','112','60','71','92','24','105','109','109','114','114','123','31','36','31','36','44','313','398','39','63','92','116','74','78','56','72','123','31','38','46','44','50','115','127','79','42','116','263','46','390','411','436','54','49','113','31','80','195','205','85','79','156','101','61','70','89','97','82','136','141','90','172','120','158','168','163','18','107','80','45','66','107','203','480','92','104','154','36','163','228','189','119','88','126','71','81','91','132','63','67','292','190','204','222','144','144','245','54','45','81','27','30','69','130','63','95','82','206','123','159','92','120','91','228','398','52','233','63','293','368','36','117','211','298','296','42','285','62','87','52','173','122','134','199','60','49','57','85','79','86','157','28','182','311','258','277','218','296','167','367','86','50','44','102','49','43','60','44','109','173','170','479','144','36','38','45','55','127','367','53']
    endsliceno=['130','187','114','277','167','64','111','124','113','188','113','86','54','89','42','39','41','125','177','111','100','42','70','91','92','100','120','160','212','220','58','113','84','77','72','72','89','108','103','181','112','63','203','117','96','69','68','102','104','154','167','81','178','122','44','143','70','128','109','201','39','37','191','25','32','53','92','38','49','57','146','68','45','64','121','120','123','126','65','56','95','95','121','41','87','91','32','68','70','94','169','55','203','104','116','55','78','58','88','34','72','35','59','115','58','48','110','44','60','85','192','223','63','206','25','180','56','55','87','87','44','135','53','196','161','54','88','313','56','56','76','31','152','40','69','76','91','187','51','448','378','157','143','148','172','111','100','61','137','69','67','83','98','175','47','450','171','211','73','156','169','47','348','149','70','84','133','65','75','96','27','113','112','115','118','122','129','40','38','40','38','61','320','404','45','72','98','140','77','84','61','89','126','41','48','53','55','63','120','131','86','46','132','272','62','401','440','442','59','53','124','67','84','201','220','87','98','160','105','72','80','106','101','88','146','147','97','190','142','166','182','171','21','112','84','48','69','111','209','510','95','106','177','39','184','248','211','130','105','135','77','84','101','147','67','70','303','195','210','231','148','150','253','56','51','94','36','41','80','135','73','107','92','212','131','161','100','155','97','238','403','58','257','69','322','382','77','124','222','310','313','56','298','72','91','56','179','127','140','207','64','70','67','91','86','101','169','43','225','330','289','290','226','306','180','390','97','55','61','107','57','50','67','63','136','186','210','493','154','38','40','51','58','132','395','66']
    totalnumberofslices=[]
    
#    lisst=['LIDC-IDRI-0001','LIDC-IDRI-0002','LIDC-IDRI-0003','LIDC-IDRI-0005','LIDC-IDRI-0006','LIDC-IDRI-0007','LIDC-IDRI-0008','LIDC-IDRI-0009','LIDC-IDRI-0011','LIDC-IDRI-0016','LIDC-IDRI-0018','LIDC-IDRI-0022','LIDC-IDRI-0023','LIDC-IDRI-0024','LIDC-IDRI-0027','LIDC-IDRI-0028']
#    for scaNumber, sca in enumerate(scan):
#        lst.append(sca.patient_id)
#    lst.sort()
  #  del lst[100:]
    lst=['LIDC-IDRI-0007','LIDC-IDRI-0018','LIDC-IDRI-0022','LIDC-IDRI-0023','LIDC-IDRI-0027']
#    lst=['LIDC-IDRI-0058','LIDC-IDRI-0095','LIDC-IDRI-0136','LIDC-IDRI-0141','LIDC-IDRI-0188','LIDC-IDRI-0190','LIDC-IDRI-0191','LIDC-IDRI-0289','LIDC-IDRI-0303','LIDC-IDRI-0325']
#    lst=['LIDC-IDRI-0027']
    lstindex=lisst.index(lst[0])
    nodNumber=nodno[lstindex]
    filename="./gtnpy/gtnpy"+lst[0]+"_"+str(nodNumber)+".npy"
    gtvol=np.load(filename,allow_pickle=True)    
#    print(lst)
    #******
#    filename="choilsnpyLIDC-IDRI-0022.npy"
#    lsvol=np.load(filename,allow_pickle=True)
#    lsvolcp=copy.copy(lsvol)
#    midsli=findMiddleslice(lsvolcp)
#    volsh=lsvolcp.shape
    #******
    startslice=0    
    for sd in range(2,6):
        for count,patid in enumerate(lst):
            pathdistance='./outputnew/'+patid+'/a2p/original/'
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patid).first()
            dcm = get_any_file(scan.get_path_to_dicom_files())
            dcmintercept = dcm.RescaleIntercept
            dcmslope = dcm.RescaleSlope
            vol = scan.to_volume()
            volcp=copy.copy(vol)
            nods = scan.cluster_annotations()  #all annotations can also be obtained as an sqlalchemy list using scan.annotations
    #        print(scan.annotations)
            lsmaskvolumebefore=np.zeros((512,512,vol.shape[2]), dtype=np.uint8)
            lsmaskvolumeafter=np.zeros((512,512,vol.shape[2]), dtype=bool)
            voldim=vol.shape
            startslice=0
            stopslice=vol.shape[2]       
    #        startslice=99
    #        stopslice=102
            noofslices=stopslice-startslice
            lsmaskvolume=np.zeros((512,512,noofslices), dtype=bool)
            #conNumber=0
            #for con in range(0,voldim[2]):
            #for con in range(cbbox[2].start,cbbox[2].stop): 
            con=0
            beginningflag=False
            ratioslist=[]
            lsmaskvolumebefore = lsmaskvolumebefore.astype(np.uint8)
            filename="./lsbefore/newlsnpy"+patid+".npy" 
            from os import path
    #        if(os.path.isfile(filename)==False):
    
            print("before creating directories")
    
                
    #        import os
            if(not os.path.isdir('./outputnew/'+patid)):
                print("before creating directories inside")
                os.makedirs('./outputnew/'+patid)
                os.makedirs('./outputnew/'+patid+'/segpng/')
                os.makedirs('./outputnew/'+patid+'/a2p/lobe/')
                os.makedirs('./outputnew/'+patid+'/a2p/hull/')
                os.makedirs('./outputnew/'+patid+'/a2p/groundtruthnodule/')
                os.makedirs('./outputnew/'+patid+'/a2p/groundtruthlobe/')
                os.makedirs('./outputnew/'+patid+'/a2p/newborder/')
                os.makedirs('./outputnew/'+patid+'/a2p/border/')
                os.makedirs('./outputnew/'+patid+'/a2p/both/')
                os.makedirs('./outputnew/'+patid+'/a2p/distance/')
                os.makedirs('./outputnew/'+patid+'/a2p/histogram/')
                os.makedirs('./outputnew/'+patid+'/a2p/rectifiedlung/')
                os.makedirs('./outputnew/'+patid+'/a2p/test/')
                os.makedirs('./outputnew/'+patid+'/a2p/onlynodules/')
                os.makedirs('./outputnew/'+patid+'/a2p/onlynodulesopencv/')
                os.makedirs('./outputnew/'+patid+'/a2p/onlynodulesopencvblob/')
                os.makedirs('./outputnew/'+patid+'/a2p/correct/')
                os.makedirs('./outputnew/'+patid+'/a2p/with_boxes/')
                os.makedirs('./outputnew/'+patid+'/a2p/intermediate_best_matches/')
                os.makedirs('./outputnew/'+patid+'/a2p/original/')
                os.makedirs('./outputnew/'+patid+'/a2p/StringNewBorder/')
                os.makedirs('./outputnew/'+patid+'/a2p/finalContour/')
                os.makedirs('./outputnew/'+patid+'/a2p/endPointUncertainities/')
                os.makedirs('./outputnew/'+patid+'/a2p/removepieces/')
                os.makedirs('./outputnew/'+patid+'/a2p/createchain/')
                os.makedirs('./outputnew/'+patid+'/a2p/loberight/')
                os.makedirs('./outputnew/'+patid+'/a2p/hullright/')
                os.makedirs('./outputnew/'+patid+'/a2p/newborderright/')
                os.makedirs('./outputnew/'+patid+'/a2p/borderright/') 
                print("before creating directories outside")
            # check if the npy file is present in lsbefore/newlsnpy folder. If it is present, you dont create it else create it.
            
            if(os.path.isfile(filename)==False):
                for i in range(startslice,stopslice):    
                   printstr=patid+" " +str(i)+"/"+str(voldim[2])
                   print(printstr)    
                   curImgCopyy=copy.copy(volcp[:,:,i])
        #           plt.figure(),plt.imshow(curImgCopyy, cmap=plt.cm.gray)
                   segmentedlung=segmentLung(curImgCopyy)
        #           plt.figure(),plt.imshow(segmentedlung, cmap=plt.cm.gray)
                   lsmaskvolumebefore[:,:,i]=segmentedlung
                   filename='./outputnew/'+patid+'/segpng/'
                   plt.imsave(filename+str(i)+"_seg.png",segmentedlung,cmap='gray')
                lsmaskvolumebefore = lsmaskvolumebefore.astype(np.uint8)
                filename="./lsbefore/newlsnpy"+patid+".npy"
                np.save(filename, lsmaskvolumebefore)  
            else:
    #        filename="./newlsnpy"+patid+".npy"
                lsmaskvolumebefore = np.load(filename,allow_pickle=True)    
            lsmaskvolumebefore = lsmaskvolumebefore.astype(np.uint8)
            lsmaskvolumebeforec = copy.copy(lsmaskvolumebefore)
            midsli,minsli,maxsli=findMiddleslice(lsmaskvolumebeforec)
          #  minsli=minsli+5
       #     maxsli=maxsli-7
    #                from os import path
            if(os.path.isfile(filename)==False):
                filename='./outputnew/'+patid+'/segpng/'
                for i in range(startslice,minsli):
                    lsmaskvolumebefore[:,:,i]=0
                    plt.imsave(filename+str(i)+"_seg.png",lsmaskvolumebefore[:,:,i],cmap='gray')
                    print(i)
                for i in range(maxsli+1,stopslice):  
                    lsmaskvolumebefore[:,:,i]=0
                    plt.imsave(filename+str(i)+"_seg.png",lsmaskvolumebefore[:,:,i],cmap='gray')
                    print(i)
               
        #    filename="./newlsnpy"+patid+".npy"
    #        lsvol=np.load(filename,allow_pickle=True)
    #        lsvolcp=copy.copy(lsvol)
        #    midsli=findMiddleslice(lsvolcp)
            lsmaskvolumebeforec=copy.copy(lsmaskvolumebefore)
            volsh=lsmaskvolumebeforec.shape
            
    
            
            
    ##        initslicenumber=99
    ##        lastslicenumber=100
    #        initslicenumber=minsli+1
    #        lastslicenumber=maxsli
    #        
    #        img1=lsvolcp[:,:,minsli]
            img1=lsmaskvolumebeforec[:,:,minsli]   
    
            slice1=copy.copy(volcp[:,:,minsli])
            slice1rgb=png2(slice1)
    #        img1=lsmaskvolumebeforec[:,:,57]    
            #plt.figure(),plt.imshow(img1, cmap=plt.cm.gray), plt.show()
    #        img1=getImgFormatted(img1)
    #        plt.figure(),plt.imshow(img1, cmap=plt.cm.gray), plt.show()
            fname="_"+str(minsli)+"_"  
    #        pathdistance='./outputnew/'+patid+'/a2p/original/'
            borderl1full,borderr1full,lobel1,lober1=getlobeborders(img1)
            pathdistance='./outputnew/'+patid+'/a2p/original/'
            cv2.imwrite(pathdistance+"orig_"+str(count)+fname+".png", slice1rgb)
    #************************************************
    #        pathborder='./outputnew/'+patid+'/a2p/test/'
            borderl1=AnteriorPosteriorDetection1(lobel1,borderl1full,slice1rgb,fname+"A2P_L1.png",patid,True)
            borderr1=AnteriorPosteriorDetection1(lober1,borderr1full,slice1rgb,fname+"A2P_R1.png",patid,False)        
    #**************************************************
    #        for i in range(58,59):
    #        for i in range(minsli+1,minsli+2):
            pathdistance='./outputnew/'+patid+'/a2p/original/'
            for i in range(minsli+1,maxsli):            
                img2=lsmaskvolumebeforec[:,:,i]
                gtimg=gtvol[:,:,i]
                #plt.figure(),plt.imshow(img2, cmap=plt.cm.gray), plt.show()            
    #            currentslice=i+startslice
                print("comparision between "+str(i-1)+"  and   "+str(i))
    #            img2=getImgFormatted(img2)
                borderl2full,borderr2full,lobel2,lober2=getlobeborders(img2)
              #  cv2.imwrite(pathdistance+"orig_"+str(i)+fname+".png", img2)
    #****************************************************************************** 
    #            score=calculatebestmatch(lobel1,lobel2)
    #            bordert1 =createnewcontour(lobel1,lobel2,score)
    #            #drawAllThreeBoundaries(borderl1,borderl2,bordert1)
    #            lengthwidth=25
    #            boxes=extractROIbestMatch(lobel1,lobel2,score,lengthwidth)
    #            
    #            fname="_"+str(i-1)+"_"+str(i)+"_"
    #            slice2=copy.copy(volcp[:,:,i])
    #            slice2rgb=png2(slice2)
    #            
    #            bordert2=borderCorrection(borderl2,borderl1,bordert1,boxes,slice2rgb,patid,fname+"BC.png",lengthwidth)
    #            borderl1=bordert1
    #            lobel1=lobel2
    #******************************************************************************            
                
                
                
    #            print(score)
    #*********************************************************************************************            
                #forleftlobe
                #1. get the pairwise distance
                #2. Compute the parameters
                #3. Check if the nodule exists
                #4. Do the Processing to locate the nodules and redraw the lung contour
                #5. Do postprocessing operations
    #            slice1=copy.copy(volcp[:,:,i-1])
                slice2=copy.copy(volcp[:,:,i])
    #            slice1rgb=png2(slice1)
                slice2rgb=png2(slice2)
                pathdistance='./outputnew/'+patid+'/a2p/original/'
                cv2.imwrite(pathdistance+"orig_"+str(i)+fname+".png", slice2rgb)
                pathborder='./outputnew/'+patid+'/a2p/test/'
                fname="_"+str(i)+"_"  
                
                borderl2=AnteriorPosteriorDetection1(lobel2,borderl2full,slice2rgb,fname+"A2P_L2.png",patid,True)
                borderr2=AnteriorPosteriorDetection1(lober2,borderr2full,slice2rgb,fname+"A2P_R2.png",patid,False) 
    #            imag=np.zeros((512,512,3),np.int8)
                fname="_"+str(i-1)+"_"+str(i)+"_"  
    #            my_dpi=96
    #            plt.figure()
    ##            plt.imshow(imag, cmap='Greys')            
    #            for pt in borderl1:
    #                imag[pt]=[0,0,0] 
    #            plt.imshow(imag, cmap='Greys')                   
    ##            cv2.imwrite(pathborder+"A2P_L1.png"+fname, imag)  
    #            plt.savefig(pathborder+fname+"A2P_L1.png", dpi=my_dpi,transparent=True)              
    #            imag=np.zeros((512,512,3),np.int8)
    #            plt.figure()
    #            for pt in borderl2:
    #                imag[pt]=[0,0,0]   
    #            plt.imshow(imag, cmap='Greys')                 
    ##            cv2.imwrite(pathborder+"A2P_L2.png"+fname, imag)  
    #            plt.savefig(pathborder+fname+"A2P_L2.png", dpi=my_dpi,transparent=True)
    #            imag=np.zeros((512,512,3),np.int8)
    #            plt.figure()
    #                
    #            for pt in borderr1:
    #                imag[pt]=[0,0,0] 
    #            plt.imshow(imag, cmap='Greys') 
    #            plt.savefig(pathborder+fname+"A2P_R1.png", dpi=my_dpi,transparent=True)  
    ##            cv2.imwrite(pathborder+"A2P_L2.png"+fname, imag)
    #            imag=np.zeros((512,512,3),np.int8)
    #            plt.figure()
    #   
    #            for pt in borderr2:
    #                imag[pt]=[0,0,0] 
    #            plt.imshow(imag, cmap='Greys')                  
    #            plt.savefig(pathborder+fname+"A2P_R2.png", dpi=my_dpi,transparent=True)  
    ##            cv2.imwrite(pathborder+"A2P_L2.png"+fname, imag)    
                
                #The information that is returned after lobeprocessing is "foundnodule or not", 
                #if foundnodule "the rectified border". This becomes reference border for next iteration
                #Is there a necessity of strides
                LY12=0
                LY21=0
                RY12=0
                RY21=0
                nodTrFl,rectifiedborderl,rectifiedlobel,LY21,LY12=LobeProcessing(borderl1,borderl2,borderl1full,borderl2full,lobel1,lobel2,slice1rgb,slice2rgb,fname+"L_lobe.png",patid,sd)
                nodTrFr,rectifiedborderr,rectifiedlober,RY21,RY12=LobeProcessing(borderr1,borderr2,borderr1full,borderr2full,lober1,lober2,slice1rgb,slice2rgb,fname+"R_lobe.png",patid,sd)
                pathdistance='./outputnew/'+patid+'/a2p/groundtruthnodule/'
                if(nodTrFl):
                    cv2.imwrite(pathdistance+"Groundtruth_Left__"+str(i)+"__"+patid+".png", gtimg)
                    gtimgb=gtimg.ravel()>0
                    
                    gtimgbin=np.reshape(gtimgb,(512,512))
    #                
                    gtimgbi=np.zeros([512, 512], dtype=np.uint8)
                 #   gtimgbi = np.arange(0, 262144, 0, np.uint8) 
                    gtimgbi[gtimgbin]=255
                    
                    cv2.imwrite(pathdistance+"Groundtruth_Left__"+str(i)+"__"+patid+".png", gtimgbi)
                    
                    gtimgl=copy.copy(gtimgbi)
                    gtimgl=np.logical_or(gtimgbi==255,lobel2==255)
                    gtimglobe=copy.copy(gtimgbi)
                    gtimglobe[gtimgl]=255
                    cv2.imwrite(pathdistance+"LobeLeft__"+str(i)+"__"+patid+".png", lobel2)
                    cv2.imwrite(pathdistance+"Groundtruth_LobeLeft__"+str(i)+"__"+patid+".png", gtimglobe)
                    
    #                gtimglb=gtimgl==255
    #                lobel2b=lobel2==255
    #                gtimglb=gtimglb or lobel2b
    #                
    #                gtimgl=gtimgl+lobel2
    #                gtimgl[gtimgl<lobel2]=255
    #                a+=b; a[a<b]=255
                    
    ##                newlobes[trufal]=0  
    ##                gtimgb = gtimgb.astype(np.uint8)
    ##                indices = np.where(gtimgb == gtimgb.max())
    ##                trufal=newlobes==127
    ##                newlobes[trufal]=0  
                    rectifiedlobelb=rectifiedlobel.ravel()>0
                    rectifiedlobelbb=copy.copy(rectifiedlobelb)
                    rectifiedlobelbb=np.reshape(rectifiedlobelb,(512,512))
                    rectifiedlobei=np.zeros([512, 512], dtype=np.uint8)
                    rectifiedlobei[rectifiedlobelbb]=255
                    rectifiedlobe_nodeb=np.logical_or(rectifiedlobei==255,lobel2==255)
                    rectifiedlobe_nodein=np.zeros([512, 512], dtype=np.uint8)
                    rectifiedlobe_nodein[rectifiedlobe_nodeb]=255
                    cv2.imwrite(pathdistance+"RectifiedNodeLeft__"+str(i)+"__"+patid+".png", rectifiedlobei)
                    cv2.imwrite(pathdistance+"RectifiedNode_LobeLeft__"+str(i)+"__"+patid+".png", rectifiedlobe_nodein)
                    
    ##                recloblngt=~np.bitwise_and(gtimgb,rectifiedlobelb)
                    
    #                pllINTgtlll=np.logical_and(gtimglobe==255,rectifiedlobe_nodein==255)
    #                pllUNIgtlll=np.logical_or(gtimglobe==255,rectifiedlobe_nodein==255)
    #                vofl=pllINTgtlll/pllUNIgtlll
    #                pllDIFFgtlll=np.logical_and(gtimglobe==0,rectifiedlobe_nodein==255)
    #                gtllDIFFplll=np.logical_and(gtimglobe==255,rectifiedlobe_nodein==0)             
    #                gtlll=count = np.count_nonzero(gtimgb)
    #                osrl=(pllDIFFgtlll)/gtlll
    #                usrl=(gtllDIFFplll)/gtlll
    
                    pllINTgtlll=np.logical_and(gtimglobe==255,rectifiedlobe_nodein==255)
                    pllINTgtllln = np.count_nonzero(pllINTgtlll)
                    pllUNIgtlll=np.logical_or(gtimglobe==255,rectifiedlobe_nodein==255)
                    pllUNIgtllln = np.count_nonzero(pllUNIgtlll)
                    vofl=pllINTgtllln/pllUNIgtllln
                    
                    pllDIFFgtlll=np.logical_and(gtimglobe==0,rectifiedlobe_nodein==255)
                    pllDIFFgtllln = np.count_nonzero(pllDIFFgtlll)
                    gtllDIFFplll=np.logical_and(gtimglobe==255,rectifiedlobe_nodein==0)             
                    gtllDIFFpllln = np.count_nonzero(gtllDIFFplll)
                    gtlll= np.count_nonzero(gtimgb)
                    try:
                        
    #                    osrl=(pllDIFFgtllln)/gtlll
    #                    usrl=(gtllDIFFpllln)/gtlll
    
                        osrl=(pllDIFFgtllln)/pllUNIgtllln
                        usrl=(gtllDIFFpllln)/pllUNIgtllln                    
                    except ZeroDivisionError:
                        print('Cannot divide by zero.')
                        osrl=999999
                        usrl=999999
    
    
                    
                    [tnl,fpl,fnl,tpl]=confusion_matrix(gtimgb,rectifiedlobelb,labels=[0,1]).ravel()
                    gtimglravel=gtimgl.ravel()
                    rectifiedlobe_nodebravel=rectifiedlobe_nodeb.ravel()
                    [tnll,fpll,fnll,tpll]=confusion_matrix(gtimglravel,rectifiedlobe_nodebravel,labels=[0,1]).ravel()
                else:
                    tnl=0
                    fpl=0
                    fnl=0
                    tpl=0
                    
                    tnll=0
                    fpll=0
                    fnll=0
                    tpll=0
                    gtlll=0
                    pllINTgtllln=0
                    pllUNIgtllln=0
                    pllDIFFgtllln=0
                    gtllDIFFpllln=0
                    vofl=0
                    gtlll=0
                    osrl=0
                    usrl=0
    
                if(nodTrFr):
                    cv2.imwrite(pathdistance+"Groundtruth_Right__"+str(i)+"__"+patid+".png", gtimg)
                    gtimgb=gtimg.ravel()>0
                    
                    gtimgbin=np.reshape(gtimgb,(512,512))
                    
                    gtimgbi=np.zeros([512, 512], dtype=np.uint8)
                 #   gtimgbi = np.arange(0, 262144, 0, np.uint8) 
                    gtimgbi[gtimgbin]=255
                    cv2.imwrite(pathdistance+"Groundtruth_Right__"+str(i)+"__"+patid+".png", gtimgbi)
    
                    gtimgl=copy.copy(gtimgbi)
                    gtimgl=np.logical_or(gtimgbi==255,lober2==255)
                    gtimglobe=copy.copy(gtimgbi)
                    gtimglobe[gtimgl]=255
                    cv2.imwrite(pathdistance+"LobeRight__"+str(i)+"__"+patid+".png", lober2)
                    cv2.imwrite(pathdistance+"Groundtruth_LobeRight__"+str(i)+"__"+patid+".png", gtimglobe)
    
    
                    
                    rectifiedloberb=rectifiedlober.ravel()>0   
                    rectifiedloberbb=copy.copy(rectifiedloberb)
                    rectifiedloberbb=np.reshape(rectifiedloberb,(512,512))
                    rectifiedlobei=np.zeros([512, 512], dtype=np.uint8)
                    rectifiedlobei[rectifiedloberbb]=255
                    rectifiedlobe_nodeb=np.logical_or(rectifiedlobei==255,lober2==255)
                    rectifiedlobe_nodein=np.zeros([512, 512], dtype=np.uint8)
                    rectifiedlobe_nodein[rectifiedlobe_nodeb]=255                
                    cv2.imwrite(pathdistance+"RectifiedNodeRight__"+str(i)+"__"+patid+".png", rectifiedlobei)
                    cv2.imwrite(pathdistance+"RectifiedNode_LobeRight__"+str(i)+"__"+patid+".png", rectifiedlobe_nodein)
                    
                    pllINTgtllr=np.logical_and(gtimglobe==255,rectifiedlobe_nodein==255)
                    pllINTgtllrn = np.count_nonzero(pllINTgtllr)
                    pllUNIgtllr=np.logical_or(gtimglobe==255,rectifiedlobe_nodein==255)
                    pllUNIgtllrn = np.count_nonzero(pllUNIgtllr)
                    vofr=pllINTgtllrn/pllUNIgtllrn
                    
                    pllDIFFgtllr=np.logical_and(gtimglobe==0,rectifiedlobe_nodein==255)
                    pllDIFFgtllrn = np.count_nonzero(pllDIFFgtllr)
                    gtllDIFFpllr=np.logical_and(gtimglobe==255,rectifiedlobe_nodein==0)             
                    gtllDIFFpllrn = np.count_nonzero(gtllDIFFpllr)
                    gtllr=count = np.count_nonzero(gtimgb)
    #                osrr=(pllDIFFgtllrn)/gtllr
    #                usrr=(gtllDIFFpllrn)/gtllr
    
                    try:
                        
    #                    osrr=(pllDIFFgtllrn)/gtllr
    #                    usrr=(gtllDIFFpllrn)/gtllr
    
                        osrr=(pllDIFFgtllrn)/pllUNIgtllrn
                        usrr=(gtllDIFFpllrn)/pllUNIgtllrn
                        
                    except ZeroDivisionError:
                        print('Cannot divide by zero.')
                        osrr=999999
                        usrr=999999
    
    #                voer=
                    
    #                reclobrngt=~np.bitwise_and(gtimgb,rectifiedloberb)   
                    [tnr,fpr,fnr,tpr]=confusion_matrix(gtimgb,rectifiedloberb,labels=[0,1]).ravel()
                    gtimglravel=gtimgl.ravel()
                    rectifiedlobe_nodebravel=rectifiedlobe_nodeb.ravel()                
                    [tnrr,fprr,fnrr,tprr]=confusion_matrix(gtimglravel,rectifiedlobe_nodebravel,labels=[0,1]).ravel()
                else:
                    tnr=0
                    fpr=0
                    fnr=0
                    tpr=0   
                    
                    tnrr=0
                    fprr=0
                    fnrr=0
                    tprr=0                
                    pllINTgtllrn=0
                    pllUNIgtllrn=0
                    pllDIFFgtllrn=0
                    gtllDIFFpllrn=0
                    vofr=0
                    gtllr=0
                    osrr=0
                    usrr=0
    
    ##        y_pred=p_mask.ravel()>0
    ##        y_true=n_mask.ravel()>0
    #        fscore=f1_score(gtimgb,rectifiedlobelb)
    #        accuracyscore=accuracy_score(y_pred,y_true)
    #        precisionscore=precision_score(y_pred,y_true)
    #        recallscore=recall_score(y_pred,y_true)
    #        jaccardscore=jaccard_score(y_pred,y_true)
    #        
    #        f1_scores+=[fscore]
    #        accuracy_scores+=[accuracyscore]
    #        precision_scores+=[precisionscore]
    #        recall_scores+=[recallscore]
    #        jaccard_scores+=[jaccardscore]
    #        
    #        featurevector.append([fscore])#tn, fp, fn, tp
    #        [tn,fp,fn,tp]=confusion_matrix(y_pred,y_true,labels=[0,1]).ravel()
    #        featurevector.append(tn)
    #        featurevector.append(fp)
    #        featurevector.append(fn)
    #        featurevector.append(tp)
    #        featurevector.append([accuracyscore])
    #        featurevector.append([precisionscore])
    #        featurevector.append([recallscore])
    #        featurevector.append([jaccardscore])
    
                borderl1full=rectifiedborderl
                borderr1full=rectifiedborderr
    #            borderr1=rectifiedborderr
                lobel1=rectifiedlobel
                lober1=rectifiedlober
    #            borderl1full=borderl1
    #            lober1=lober2
                slice1=slice2
                slice1rgb=slice2rgb
                borderl1=AnteriorPosteriorDetection1(lobel1,borderl1full,slice1rgb,fname+"NN_A2P_L1.png",patid,True)
                borderr1=AnteriorPosteriorDetection1(lober1,borderr1full,slice1rgb,fname+"NN_A2P_R1.png",patid, False)
                featurevector = [] 
                featurevector.append(patid)
                featurevector.append(i-1)
                featurevector.append(i)
                featurevector.append(nodTrFl)
                featurevector.append(LY12)
                featurevector.append(LY21)
                featurevector.append(RY12)
                featurevector.append(RY21)
                
                
    #            featurevector.append("False")
                featurevector.append(nodTrFr)
                
                featurevector.append(tnl)
                featurevector.append(fpl)
                featurevector.append(fnl)
                featurevector.append(tpl)
                featurevector.append(tnr)
                featurevector.append(fpr)
                featurevector.append(fnr)
                featurevector.append(tpr) 
                
                featurevector.append(tnll)
                featurevector.append(fpll)
                featurevector.append(fnll)
                featurevector.append(tpll)
                featurevector.append(tnrr)
                featurevector.append(fprr)
                featurevector.append(fnrr)
                featurevector.append(tprr)  
                featurevector.append(pllINTgtllln)
                featurevector.append(pllUNIgtllln)
                featurevector.append(pllDIFFgtllln)
                featurevector.append(gtllDIFFpllln)
                featurevector.append(vofl)
                featurevector.append(gtlll)
                featurevector.append(osrl)
                featurevector.append(usrl)
                featurevector.append(pllINTgtllrn)
                featurevector.append(pllUNIgtllrn)
                featurevector.append(pllDIFFgtllrn)
                featurevector.append(gtllDIFFpllrn)
                featurevector.append(vofr)
                featurevector.append(gtllr)
                featurevector.append(osrr)
                featurevector.append(usrr)                 
                df_temp = pd.DataFrame([featurevector],columns=names)
                df=df.append(df_temp)
        df.to_csv("./outputnew/resultsx_"+patid+"_"+str(sd)+".csv") # /'+patid+'/             
            #include code for lobe assignment also lobel1=lobel2
            #lober1=lober2
##            forrightlobe
            

# ********************************************************************************************           
            
            

                
               
    
if(__name__=="__main__"):
    main()  


