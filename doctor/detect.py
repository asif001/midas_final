import numpy as np
import os
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras.models import *
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import *
# noinspection PyUnresolvedReferences
from tensorflow.keras.optimizers import *
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# noinspection PyUnresolvedReferences
from tensorflow.keras.applications.vgg16 import VGG16
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers
# noinspection PyUnresolvedReferences
from tensorflow.keras import backend as keras

import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage import measure
import pandas as pd
import pickle
import skimage.measure
from scipy.signal import find_peaks
import cv2
import math
from skimage.feature import hog

import os
import csv
import random
from skimage.transform import resize
# noinspection PyUnresolvedReferences
import matplotlib.patches as patches

# In[ ]:


# In[ ]:


weight_dir = "./modelweight/"

"""
Task : View Classification of chest X-ray image - PA vs AP
Authors : Sakib Reza & Ohida-Binte-Amin

This is an orginal research outcome
from a part of  my final year thesis work 
supervised by 
Dr. M.M.A. Hashem, Professor, Dept. of CSE, KUET.

"""

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
    fd, hog_image = hog(filt, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                        multichannel=False)

    final_feat = np.reshape(np.append(x, y), (1, 128))
    hogfeat = np.reshape(fd, (1, 128))
    final_feat = np.reshape(np.append(final_feat, hogfeat), (1, 256))
    feat_df = pd.DataFrame(final_feat)

    feat1 = np.array(feat_df.iloc[0, 2:62])
    peaks = find_peaks(feat1)[0]

    if peaks.shape[0] < 1:
        feat_df['p1'] = 16
    else:
        feat_df['p1'] = peaks[0]  # f1

    if peaks.shape[0] < 2:
        feat_df['p2'] = 32
    else:
        feat_df['p2'] = peaks[1]  # f2

    if peaks.shape[0] < 3:
        feat_df['p3'] = (64 - feat_df['p2']) // 2
    else:
        feat_df['p3'] = peaks[2]  # f3

    feat_df['ph1'] = feat1[feat_df['p1']]  # f4
    feat_df['ph2'] = feat1[feat_df['p2']]  # f5
    feat_df['ph3'] = feat1[feat_df['p3']]  # f6

    feat_df['pdx12'] = math.fabs(feat_df['p1'][0] - feat_df['p2'][0])  # f7
    feat_df['pdx23'] = math.fabs(feat_df['p2'][0] - feat_df['p3'][0])  # f8
    feat_df['pdx31'] = math.fabs(feat_df['p3'][0] - feat_df['p1'][0])  # f9
    feat_df['hdy12'] = math.fabs(feat_df['ph1'][0] - feat_df['ph2'][0])  # f10
    feat_df['hdy23'] = math.fabs(feat_df['ph2'][0] - feat_df['ph3'][0])  # f11
    feat_df['hdy_avg'] = math.fabs(feat_df['ph2'][0] - (feat_df['ph1'][0] + feat_df['ph3'][0]) / 2.0)  # f12

    notch = find_peaks(-feat1)[0]

    if notch.shape[0] < 1:
        feat_df['n1'] = (feat_df['p1'] + feat_df['p2']) // 2
    else:
        feat_df['n1'] = notch[0]  # f13

    if notch.shape[0] < 2:
        feat_df['n2'] = 59 - feat_df['n1']
    else:
        feat_df['n2'] = notch[1]  # f14

    feat_df['nh1'] = feat1[feat_df['n1']]  # f15
    feat_df['nh2'] = feat1[feat_df['n2']]  # f16
    feat_df['ndx'] = math.fabs(feat_df['n1'][0] - feat_df['n2'][0])  # f17
    feat_df['ndy'] = math.fabs(feat_df['nh1'][0] - feat_df['nh2'][0])  # f18
    feat_df['n1_p1_dx'] = math.fabs(feat_df['p1'][0] - feat_df['n1'][0])  # f19
    feat_df['n1_p2_dx'] = math.fabs(feat_df['p2'][0] - feat_df['n1'][0])  # f20
    feat_df['n1_p3_dx'] = math.fabs(feat_df['p3'][0] - feat_df['n1'][0])  # f21
    feat_df['n2_p1_dx'] = math.fabs(feat_df['p1'][0] - feat_df['n2'][0])  # f22
    feat_df['n2_p2_dx'] = math.fabs(feat_df['p2'][0] - feat_df['n2'][0])  # f23
    feat_df['n2_p3_dx'] = math.fabs(feat_df['p3'][0] - feat_df['n2'][0])  # f24
    feat_df['n1_p1_dy'] = math.fabs(feat_df['ph1'][0] - feat_df['nh1'][0])  # f25
    feat_df['n1_p2_dy'] = math.fabs(feat_df['ph2'][0] - feat_df['nh1'][0])  # f26
    feat_df['n1_p3_dy'] = math.fabs(feat_df['ph3'][0] - feat_df['nh1'][0])  # f27
    feat_df['n2_p1_dy'] = math.fabs(feat_df['ph1'][0] - feat_df['nh2'][0])  # f28
    feat_df['n2_p2_dy'] = math.fabs(feat_df['ph2'][0] - feat_df['nh2'][0])  # f29
    feat_df['n2_p3_dy'] = math.fabs(feat_df['ph3'][0] - feat_df['nh2'][0])  # f30

    return feat_df


"""
Function for directly predicting the view label - PA or AP from an input image

file : the input image file 
saved_model : the pretrained view classifier model 
label : the predicted view label - PA or AP

"""

# testing the view classifer
# result = view_predict('image_1.png')
# print(result)


# In[ ]:


# In[ ]:


def res_path(inputs, filter_size, path_number):
    def block(x, fl):
        # noinspection PyUnresolvedReferences
        cnn1 = Conv2D(filter_size, (3, 3), padding='same', activation="relu")(inputs)
        # noinspection PyUnresolvedReferences
        cnn2 = Conv2D(filter_size, (1, 1), padding='same', activation="relu")(inputs)

        add = layers.Add()([cnn1, cnn2])

        return add

    cnn = block(inputs, filter_size)
    if path_number <= 3:
        cnn = block(cnn, filter_size)
        if path_number <= 2:
            cnn = block(cnn, filter_size)
            if path_number <= 1:
                cnn = block(cnn, filter_size)

    return cnn


def dice_coef(y_true, y_pred):
    # noinspection PyUnresolvedReferences
    y_true_f = keras.flatten(y_true)
    # noinspection PyUnresolvedReferences
    y_pred_f = keras.flatten(y_pred)
    # noinspection PyUnresolvedReferences
    intersection = keras.sum(y_true_f * y_pred_f)
    # noinspection PyUnresolvedReferences
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


"""
A Keras implementation of TernausNet16: 
https://arxiv.org/abs/1801.05746
https://github.com/ternaus/TernausNet
The architecture is very similar to the original U-net paper:
https://arxiv.org/abs/1505.04597
The key differences are:
- A VGG16 architecture is used for encoder, pretrained on ImageNet
- No batchnorm used
- No dropout used
- Shortcut concatenations are mismatched on number of filters meaning that 
  a larger number of filters is used in decoder.
"""


def decoder_block_ternausV2(inputs, mid_channels, out_channels):
    """
    Decoder block as proposed for TernausNet16 here:
    https://arxiv.org/abs/1801.05746
    See DecoderBlockV2 here:
    https://github.com/ternaus/TernausNet/blob/master/unet_models.py
    - Concatenate u-net shortcut to input pre-upsample
    - Bilinear upsample input to double Height and Width dimensions
    - Note: The original ternausNet implementation includes option for
      deconvolution instead of bilinear upsampling. Omitted here because I
      couldn't find a meaningful performance comparison
    """

    conv_kwargs = dict(
        activation='relu',
        padding='same',
        kernel_initializer='he_normal',
        data_format='channels_last'  # (batch, height, width, channels)
    )

    # noinspection PyUnresolvedReferences
    x = UpSampling2D(size=(2, 2))(inputs)  # interpolation='bilinear' doesn't work?
    # noinspection PyUnresolvedReferences
    x = Conv2D(mid_channels, 3, **conv_kwargs)(x)
    # noinspection PyUnresolvedReferences
    x = Conv2D(out_channels, 3, **conv_kwargs)(x)
    return x


# INTENDED API
# ------------------------------------------------------------------------------

def TransResUNet(input_size=(512, 512, 1), output_channels=1):
    """
    A Keras implementation of TernausNet16:
    https://arxiv.org/abs/1801.05746
    https://github.com/ternaus/TernausNet
    """

    # input
    # convert 1 channel grayscale to 3 channels if needed
    # noinspection PyUnresolvedReferences
    inputs = Input(input_size)
    if input_size[-1] < 3:
        # noinspection PyUnresolvedReferences
        x = Conv2D(3, 1)(inputs)  # add channels
        input_shape = (input_size[0], input_size[0], 3)  # update input size
    else:
        x = inputs
        input_shape = input_size

    # Load pretrained VGG, conv layers include relu activation
    encoder = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # (None, 256, 256, 3)
    e1 = encoder.get_layer(name='block1_conv1')(x)
    e1 = encoder.get_layer(name='block1_conv2')(e1)
    # (None, 256, 256, 64)
    # noinspection PyUnresolvedReferences
    e2 = MaxPooling2D(pool_size=(2, 2))(e1)
    e2 = encoder.get_layer(name='block2_conv1')(e2)
    e2 = encoder.get_layer(name='block2_conv2')(e2)
    # (None, 128, 128, 128)
    # noinspection PyUnresolvedReferences
    e3 = MaxPooling2D(pool_size=(2, 2))(e2)
    e3 = encoder.get_layer(name='block3_conv1')(e3)
    e3 = encoder.get_layer(name='block3_conv2')(e3)
    e3 = encoder.get_layer(name='block3_conv3')(e3)
    # (None, 64, 64, 256)
    #     e4 = MaxPooling2D(pool_size=(2, 2))(e3)
    #     e4 = encoder.get_layer(name='block4_conv1')(e4)
    #     e4 = encoder.get_layer(name='block4_conv2')(e4)
    #     e4 = encoder.get_layer(name='block4_conv3')(e4)
    # (None, 32, 32, 512)
    #     e5 = MaxPooling2D(pool_size=(2, 2))(e4)
    #     e5 = encoder.get_layer(name='block5_conv1')(e5)
    #     e5 = encoder.get_layer(name='block5_conv2')(e5)
    #     e5 = encoder.get_layer(name='block5_conv3')(e5)
    # (None, 16, 16, 512)
    # noinspection PyUnresolvedReferences
    center = MaxPooling2D(pool_size=(2, 2))(e3)
    # (None, 8, 8, 512)
    center = decoder_block_ternausV2(center, 512, 256)
    # (None, 16, 16, 256)
    #     d5 = concatenate([e5, center], axis=3)
    #     d5 = decoder_block_ternausV2(d5, 512, 256)
    #     # (None, 32, 32, 256)
    #     res_path4 = res_path(e4,256,4)
    #     d4 = concatenate([res_path4, center], axis=3)
    #     d4 = decoder_block_ternausV2(d4, 512, 128)
    # (None, 64, 64, 128)
    res_path3 = res_path(e3, 128, 3)
    # noinspection PyUnresolvedReferences
    d3 = concatenate([res_path3, center], axis=3)
    d3 = decoder_block_ternausV2(d3, 256, 64)
    # (None, 128, 128, 64)
    res_path2 = res_path(e2, 64, 2)
    # noinspection PyUnresolvedReferences
    d2 = concatenate([res_path2, d3], axis=3)
    d2 = decoder_block_ternausV2(d2, 128, 64)
    # (None, 256, 256, 64)
    # Note: no decoder block used at end
    res_path1 = res_path(e1, 32, 1)
    # noinspection PyUnresolvedReferences
    d1 = concatenate([res_path1, d2], axis=3)
    # noinspection PyUnresolvedReferences
    d1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(d1)
    # noinspection PyUnresolvedReferences
    d1 = ReLU()(d1)
    # (None, 256, 256, 32)

    # Output
    if output_channels > 1:
        # untested
        # noinspection PyUnresolvedReferences
        op = tf.nn.log_softmax_v2(d1, axis=3)
    else:
        # noinspection PyUnresolvedReferences
        op = Conv2D(output_channels, 1)(d1)
        # noinspection PyUnresolvedReferences
        op = Activation('sigmoid')(op)  # note: ternaus excludes

    # Build
    # noinspection PyUnresolvedReferences
    model = Model(inputs=[inputs], outputs=[op])
    return model


lung_model = TransResUNet()
# noinspection PyUnresolvedReferences
lung_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])

lung_model.load_weights(weight_dir + 'PA_seg.hdf5')


# In[300]:


def lung_mask(img, thresh=10000):
    view = 'AP'
    lungs_box = []
    img = cv2.resize(img, (512, 512))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    if view == 'PA':
        mask = lung_model.predict(np.reshape(img, (1, 512, 512, 1)))
    else:
        mask = lung_model.predict(np.reshape(img, (1, 512, 512, 1)))
    mask = mask[:, :, :, 0]
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    comp = measure.label(mask[0, :, :])
    img *= 255
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    comp = measure.label(comp)
    # apply bounding boxes
    predictionString = ''
    for region in measure.regionprops(comp):
        # retrieve x, y, height and width
        y1, x1, y2, x2 = region.bbox
        height = y2 - y1
        width = x2 - x1
        if height * width < thresh:
            continue
        lungs_box.append((x1, y1, x2, y2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    cv2.imwrite('lung.jpg', img)
    lungs_box.sort()
    return (mask[0, :, :], lungs_box)


from tensorflow import keras


def create_downsample(channels, inputs):
    # noinspection PyUnresolvedReferences
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    # noinspection PyUnresolvedReferences
    x = keras.layers.LeakyReLU(0)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.MaxPool2D(2)(x)
    return x


def create_resblock(channels, inputs):
    # noinspection PyUnresolvedReferences
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    # noinspection PyUnresolvedReferences
    x = keras.layers.LeakyReLU(0)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.LeakyReLU(0)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    # noinspection PyUnresolvedReferences
    return keras.layers.add([x, inputs])


def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    # noinspection PyUnresolvedReferences
    inputs = keras.Input(shape=(input_size, input_size, 1))
    # noinspection PyUnresolvedReferences
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    # noinspection PyUnresolvedReferences
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.LeakyReLU(0)(x)
    # noinspection PyUnresolvedReferences
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    # noinspection PyUnresolvedReferences
    outputs = keras.layers.UpSampling2D(2 ** depth)(x)
    # noinspection PyUnresolvedReferences
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# In[303]:

# noinspection PyUnresolvedReferences
from tensorflow.keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = (true_positives + 1.) / (possible_positives + 1.)
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = (true_positives + 1.) / (predicted_positives + 1.)
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    # noinspection PyUnresolvedReferences
    y_true = tf.reshape(y_true, [-1])
    # noinspection PyUnresolvedReferences
    y_pred = tf.reshape(y_pred, [-1])
    # noinspection PyUnresolvedReferences
    intersection = tf.reduce_sum(y_true * y_pred)
    # noinspection PyUnresolvedReferences
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score


# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    # noinspection PyUnresolvedReferences
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)


# mean iou as a metric
def mean_iou(y_true, y_pred):
    # noinspection PyUnresolvedReferences
    y_pred = tf.round(y_pred)
    # noinspection PyUnresolvedReferences
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    # noinspection PyUnresolvedReferences
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    # noinspection PyUnresolvedReferences
    smooth = tf.ones(tf.shape(intersect))
    # noinspection PyUnresolvedReferences
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))


# create network and compiler
pneum_model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)
pneum_model.compile(optimizer='adam',
                    loss=iou_bce_loss,
                    metrics=['accuracy', recall_m, precision_m, f1_m, mean_iou])
pneum_model.load_weights(weight_dir + 'PA_FCN_model.h5')


# In[304]:


def GetPosition(pneum_box, lungs_box):
    cx = min(pneum_box[0], pneum_box[2]) + abs(pneum_box[0] - pneum_box[2]) // 2
    cy = min(pneum_box[1], pneum_box[3]) + abs(pneum_box[1] - pneum_box[3]) // 2

    #     print(cx,cy)
    lx1, ly1, lx2, ly2 = lungs_box[0][0], lungs_box[0][1], lungs_box[0][2], lungs_box[0][3]
    rx1, ry1, rx2, ry2 = lungs_box[1][0], lungs_box[1][1], lungs_box[1][2], lungs_box[1][3]
    #     print((cx,cy),lungs_box)
    if (lx1 <= cx and cx <= lx2) and (ly1 <= cy and cy <= ly2 // 2):
        return ('upper', 'left')
    elif (lx1 <= cx and cx <= lx2) and (ly1 // 2 <= cy and cy <= ly2):
        return ('lower', 'left')
    elif (rx1 <= cx and cx <= rx2) and (ry1 <= cy and cy <= ry2 // 2):
        return ('upper', 'right')
    else:
        return ('lower', 'right')


def report_generator(pneum_box, lungs_box):
    Report = 'Pneumonia evidence found in '
    st = set()
    for box in pneum_box:
        st.add(GetPosition(box, lungs_box))
    #         print(GetPosition(box, lungs_box))
    if len(st) == 0:
        Report = 'No pneumonia evidence found'
        return Report

    cnt = 0
    #     print(st)
    for i in st:

        if cnt == len(st) - 1 and len(st) != 1:
            Report += ' and '
        elif cnt > 0:
            Report += ', '
        Report += i[0] + ' part of ' + i[1] + ' lung'
        cnt += 1
    return Report


# Report = report_generator(pneum_box, lungs_box)
# print('Report: '+Report)


# In[305]:


def detect_pneum(img, img1, thresh=1500):
    view = 'AP'
    pneum_box = []
    img = cv2.resize(img, (256, 256))
    img = img / 255
    test = np.reshape(img, (1, 256, 256, 1))
    if view == 'AP':
        pred = pneum_model.predict(test)
    else:
        pred = pneum_model.predict(test)
    pred[pred <= 0.5] = 0
    pneum_conf = np.mean(pred[pred > 0.5])
    comp = pred > 0.5
    comp = np.reshape(comp, (256, 256))
    comp = cv2.resize(np.float32(comp), (512, 512))
    lung_ret = lung_mask(img1)
    mask = lung_ret[0]
    lungs_box = lung_ret[1]
    comp[comp > 0.5] = 1
    comp[comp <= 0.5] = 0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    intersect = cv2.bitwise_and(comp, mask, mask=None)
    intersect = np.int8(intersect)
    img = cv2.resize(img, (512, 512))
    img *= 255
    img = np.float32(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    comp = measure.label(intersect)
    # apply bounding boxes
    predictionString = ''
    for region in measure.regionprops(comp):
        # retrieve x, y, height and width
        y1, x1, y2, x2 = region.bbox
        height = y2 - y1
        width = x2 - x1
        if height * width < thresh:
            continue
        pneum_box.append((x1, y1, x2, y2))
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
    pneum_box.sort(key=lambda tup: tup[0])
    report = 'report : ' + report_generator(pneum_box, lungs_box)

    text_file = open("Report.txt", "w")

    text_file.write('pneumonia confidence: ' + str(pneum_conf * 100)[0:5] + '\n' + report)

    text_file.close()

    cv2.imwrite('pneum.jpg', img)
    return ('pneumonia confidence: ' + str(pneum_conf * 100)[0:5], report)


# In[306]:


# In[307]:


# In[308]:


# In[309]:


def ultimo(img_pneum, img_lung):
    mask = lung_mask(img_lung)[0]
    lung_out = cv2.imread('lung.jpg')
    report = detect_pneum(img_pneum, img_lung)
    pneum_out = cv2.imread('pneum.jpg')

    return {'pneum': pneum_out, 'lung': lung_out, 'report': report}
