# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 23:58:48 2019

@author: Dell
"""

import cv2
img1 = cv2.imread('benign1.jpg')
#img1 = cv2.rectangle(img1,(364, 670),(464, 808),(0,255,0),3)
#img1 = cv2.rectangle(img1,(671, 423),(893, 759),(0,255,0),3)
cv2.imshow('hello',img1)