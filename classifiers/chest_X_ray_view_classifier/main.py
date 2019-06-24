"""
Task : View Classification of chest X-ray image - PA vs AP
Author : Sakib Reza

This is an orginal research outcome
from a part of  my final year thesis work 
supervised by 
Dr. M.M.A. Hashem, Professor, Dept. of CSE, KUET.

"""
import numpy as np
import pandas as pd
import pickle
import skimage.measure
from scipy.signal import find_peaks
import cv2
import math
from skimage.feature import hog
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_ray_classifier_dir = os.path.join(BASE_DIR, "chest_X_ray_view_classifier")

np.warnings.filterwarnings('ignore')

"""
function for extracting feature from an image 

file : the input image file
feat_df : the panda dataframe containing the features extracted from the image 

"""


def extract_feature(file):
    img = cv2.imread(file, 0)
    pool = skimage.measure.block_reduce(img, (16, 16), np.mean)
    filt = cv2.GaussianBlur(pool, (5, 5), 0)
    vip = np.mean(filt, axis=0)
    hip = np.mean(filt, axis=1)
    x = np.reshape(vip, (1, 64))
    y = np.reshape(hip, (1, 64))
    fd, hog_image = hog(filt, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, multichannel=False)

    final_feat = np.reshape(np.append(x, y), (1, 128))
    hogfeat = np.reshape(fd, (1, 128))
    final_feat = np.reshape(np.append(final_feat, hogfeat), (1, 256))
    feat_df = pd.DataFrame(final_feat)
    
    feat1 = np.array(feat_df.iloc[0,2:62])
    peaks = find_peaks(feat1)[0]

    if peaks.shape[0]<1:
        feat_df['p1'] = 16
    else:    
        feat_df['p1'] = peaks[0] #f1

    if peaks.shape[0]<2:
        feat_df['p2'] = 32
    else:    
        feat_df['p2'] = peaks[1] #f2

    if peaks.shape[0] < 3: 
        feat_df['p3'] = (64-feat_df['p2'])//2
    else:    
        feat_df['p3'] = peaks[2] #f3 
    
    feat_df['ph1'] = feat1[feat_df['p1']] #f4
    feat_df['ph2'] = feat1[feat_df['p2']] #f5
    feat_df['ph3'] = feat1[feat_df['p3']] #f6
    
    feat_df['pdx12'] = math.fabs(feat_df['p1'][0] - feat_df['p2'][0]) #f7
    feat_df['pdx23'] = math.fabs(feat_df['p2'][0] - feat_df['p3'][0]) #f8
    feat_df['pdx31'] = math.fabs(feat_df['p3'][0] - feat_df['p1'][0]) #f9
    feat_df['hdy12'] = math.fabs(feat_df['ph1'][0] - feat_df['ph2'][0]) #f10
    feat_df['hdy23'] = math.fabs(feat_df['ph2'][0] - feat_df['ph3'][0]) #f11
    feat_df['hdy_avg'] = math.fabs(feat_df['ph2'][0] - (feat_df['ph1'][0]+feat_df['ph3'][0])/2.0 ) #f12

    notch = find_peaks(-feat1)[0]

    if notch.shape[0]<1:
        feat_df['n1'] = (feat_df['p1'] + feat_df['p2'])//2
    else:    
        feat_df['n1'] = notch[0] #f13

    if notch.shape[0]<2:
        feat_df['n2'] = 59 - feat_df['n1']
    else:    
        feat_df['n2'] = notch[1] #f14

    feat_df['nh1'] = feat1[feat_df['n1']] #f15
    feat_df['nh2'] = feat1[feat_df['n2']] #f16
    feat_df['ndx'] = math.fabs(feat_df['n1'][0] - feat_df['n2'][0]) #f17
    feat_df['ndy'] = math.fabs(feat_df['nh1'][0] - feat_df['nh2'][0]) #f18 
    feat_df['n1_p1_dx'] = math.fabs(feat_df['p1'][0] - feat_df['n1'][0]) #f19
    feat_df['n1_p2_dx'] = math.fabs(feat_df['p2'][0] - feat_df['n1'][0]) #f20
    feat_df['n1_p3_dx'] = math.fabs(feat_df['p3'][0] - feat_df['n1'][0]) #f21
    feat_df['n2_p1_dx'] = math.fabs(feat_df['p1'][0] - feat_df['n2'][0]) #f22
    feat_df['n2_p2_dx'] = math.fabs(feat_df['p2'][0] - feat_df['n2'][0]) #f23
    feat_df['n2_p3_dx'] = math.fabs(feat_df['p3'][0] - feat_df['n2'][0]) #f24
    feat_df['n1_p1_dy'] = math.fabs(feat_df['ph1'][0] - feat_df['nh1'][0]) #f25
    feat_df['n1_p2_dy'] = math.fabs(feat_df['ph2'][0] - feat_df['nh1'][0]) #f26
    feat_df['n1_p3_dy'] = math.fabs(feat_df['ph3'][0] - feat_df['nh1'][0]) #f27
    feat_df['n2_p1_dy'] = math.fabs(feat_df['ph1'][0] - feat_df['nh2'][0]) #f28
    feat_df['n2_p2_dy'] = math.fabs(feat_df['ph2'][0] - feat_df['nh2'][0]) #f29
    feat_df['n2_p3_dy'] = math.fabs(feat_df['ph3'][0] - feat_df['nh2'][0]) #f30
    
    return feat_df


"""
Function for directly predicting the view label - PA or AP from an input image

file : the input image file 
saved_model : the pretrained view classifier model 
label : the predicted view label - PA or AP

"""


def view_predict(file):


    df = extract_feature(file)
    
    feat = [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0]
    feat_ls = []
    for i in range(len(feat)):
        if feat[i] == 1:
            feat_ls.append(i)
            
    X = df.iloc[:, feat_ls].values
    
    saved_model = 'view_model.sav'
    clf = pickle.load(open(os.path.join(X_ray_classifier_dir, saved_model), 'rb'))
    prediction = clf.predict(X)[0]
    
    label = ''
    if prediction == 0:
        label = 'PA'
    else:
        label = 'AP'
    return label
