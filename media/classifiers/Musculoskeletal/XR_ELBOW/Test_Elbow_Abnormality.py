# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 16:13:04 2019

@author: prant
"""

from keras.models import load_model
import cv2
import numpy as np
from keras import optimizers

model = load_model('Elbow_VGG_Model_by_Machine.h5')
model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"]) 

## Normal X-RAY

test_image = cv2.imread('normal_XR_ELBOW_hist_13.png')
test_image = cv2.resize(test_image,(256,256))
test_image = np.reshape(test_image,[1,256,256,3])

classifier_result = model.predict(test_image)

# Predicted Class Label Selection
if(classifier_result[0][0] >= classifier_result[0][1]):
    print("\nX-RAY is Abnormal")
    class_label = 0
elif(classifier_result[0][0] < classifier_result[0][1]):
    print("\nX-RAY is Normal")
    class_label = 1



## Abnormal X-RAY
    
test_image = cv2.imread('abnormal_XR_ELBOW_hist_276.png')
test_image = cv2.resize(test_image,(256,256))
test_image = np.reshape(test_image,[1,256,256,3])

classifier_result = model.predict(test_image)

# Predicted Class Label Selection
if(classifier_result[0][0] >= classifier_result[0][1]):
    print("\nX-RAY is Abnormal")
    class_label = 0
elif(classifier_result[0][0] < classifier_result[0][1]):
    print("\nX-RAY is Normal")
    class_label = 1