#----------------------------------------------------------------------------
#  Author       : Atheeq Basha Syed
#  Student ID   : 700041541
#  File Name    : prog_help.py
#  Project Name : Underwater Video Analysis for Fish Biodiversity Monitoring
#  Description  : Implementation of EfficientNet and YOLO models
#                 on fish species dataset
#----------------------------------------------------------------------------

# Required Imports to execute the YOLO and EfficientNet Tasks
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import colorsys
import json
import pandas as pd
import shutil
import random
import os
from tqdm import tqdm
from sys import exit
import argparse
from textwrap import dedent
from lxml import etree
import xml.etree.ElementTree as ET
import glob
import matplotlib
import tensorflow.keras as keras
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
from efficientnet.tfkeras import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from efficientnet.tfkeras import EfficientNetB3
from tensorflow.keras.models import load_model
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Layer, BatchNormalization, GlobalAveragePooling2D 
from tensorflow.keras.layers import Input, LeakyReLU, ZeroPadding2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import l2
from multiprocessing import Process, Queue, Pipe
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import auc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow.python.client import device_lib

# Options to train and test the YOLO model
yolo_weights                = "dataset/yolov3.weights"
yolo_custom_weights         = False 
coco_classes                = "dataset/coco.names"
stri                        = np.array([8, 16, 32])  #YOLO strides
input_size                  = 416
anchor                      = np.array([[[10,  13], [16,   30], [33,   23]], 
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]])
anchor                      = (np.array(anchor).T/stri).T # Anchors used

# Options to train and Test 
save_best                   = True 
save_checkpoint             = False 
train_classes               = "dataset/fish_detection_names.txt"
train_annots_path           = "dataset/fish_detection_train.txt"
test_annots_path            = "dataset/fish_detection_test.txt"
log_dir                     = "log"
checkpoints_path            = "checkpoints"
model_name                  = "yolov3"
load_from_RAM               = True 
batch_size                  = 4
train_data_aug              = True
test_data_aug               = False
transfer                    = True
train_from_checkpoint       = False 
epochs_total                = 20

# This section belongs to Dataset Download

def label_to_xml(Dataset_path):
    '''
    Input: The path where Dataset is Downloaded
    Output: XML files for the Images in Dataset
    Function: Creates Pascal VOC format of XML file
    from the annotations of images
    '''
    os.chdir(Dataset_path)
    DIRS = os.listdir(os.getcwd())

    for DIR in DIRS:
        if os.path.isdir(DIR):
            os.chdir(DIR)

            print("Currently in Subdirectory:", DIR)
            CLASS_DIRS = os.listdir(os.getcwd()) 
            for CLASS_DIR in CLASS_DIRS:
                if " " in CLASS_DIR:
                    os.rename(CLASS_DIR, CLASS_DIR.replace(" ", "_"))
            
            CLASS_DIRS = os.listdir(os.getcwd())
            for CLASS_DIR in CLASS_DIRS:
                if os.path.isdir(CLASS_DIR):
                    os.chdir(CLASS_DIR)

                    print("\n" + "Creating XML Files for Class:", CLASS_DIR)

                    os.chdir("labels")

                    for filename in tqdm(os.listdir(os.getcwd())):
                        if filename.endswith(".txt"):
                            filename_str = str.split(filename, ".")[0]


                            annotation = etree.Element("annotation")
                            
                            os.chdir("..")
                            folder = etree.Element("folder")
                            folder.text = os.path.basename(os.getcwd())
                            annotation.append(folder)

                            filename_xml = etree.Element("filename")
                            filename_xml.text = filename_str + ".jpg"
                            annotation.append(filename_xml)

                            path = etree.Element("path")
                            path.text = os.path.join(os.path.dirname(os.path.abspath(filename)), filename_str + ".jpg")
                            annotation.append(path)

                            source = etree.Element("source")
                            annotation.append(source)

                            database = etree.Element("database")
                            database.text = "Unknown"
                            source.append(database)

                            size = etree.Element("size")
                            annotation.append(size)

                            width = etree.Element("width")
                            height = etree.Element("height")
                            depth = etree.Element("depth")

                            img = cv2.imread(filename_xml.text)

                            try:
                                width.text = str(img.shape[1])
                            except AttributeError:
                                os.chdir("labels")
                                continue
                            height.text = str(img.shape[0])
                            depth.text = str(img.shape[2])

                            size.append(width)
                            size.append(height)
                            size.append(depth)

                            segmented = etree.Element("segmented")
                            segmented.text = "0"
                            annotation.append(segmented)

                            os.chdir("labels")
                            label_original = open(filename, 'r')

                            for line in label_original:
                                line = line.strip()
                                l = line.split(' ')
                                
                                class_name_len = len(l) - 4 # 4 coordinates
                                class_name = l[0]
                                for i in range(1,class_name_len):
                                    class_name = f"{class_name}_{l[i]}"

                                addi = class_name_len

                                xmin_l = str(int(round(float(l[0+addi]))))
                                ymin_l = str(int(round(float(l[1+addi]))))
                                xmax_l = str(int(round(float(l[2+addi]))))
                                ymax_l = str(int(round(float(l[3+addi]))))
                                
                                obj = etree.Element("object")
                                annotation.append(obj)

                                name = etree.Element("name")
                                name.text = class_name
                                obj.append(name)

                                pose = etree.Element("pose")
                                pose.text = "Unspecified"
                                obj.append(pose)

                                truncated = etree.Element("truncated")
                                truncated.text = "0"
                                obj.append(truncated)

                                difficult = etree.Element("difficult")
                                difficult.text = "0"
                                obj.append(difficult)

                                bndbox = etree.Element("bndbox")
                                obj.append(bndbox)

                                xmin = etree.Element("xmin")
                                xmin.text = xmin_l
                                bndbox.append(xmin)

                                ymin = etree.Element("ymin")
                                ymin.text = ymin_l
                                bndbox.append(ymin)

                                xmax = etree.Element("xmax")
                                xmax.text = xmax_l
                                bndbox.append(xmax)

                                ymax = etree.Element("ymax")
                                ymax.text = ymax_l
                                bndbox.append(ymax)

                            os.chdir("..")
                            s = etree.tostring(annotation, pretty_print=True)
                            with open(filename_str + ".xml", 'wb') as f:
                                f.write(s)
                                f.close()

                            os.chdir("labels")

                    os.chdir("..")
                    os.chdir("..")   
                       
            os.chdir("..")

def xml_to_txt(train_path, test_path, names_path, subfolder=True):
    '''
    Input: The train, test and names path of Dataset
    Output: Text files with image path and annotations
    Function: Creates text file from XML file with the image
    path and annotations
    '''
    names = []
    for i, folder in enumerate(['train','test']):
        with open([train_path,test_path][i], "w") as file:
            data_path = os.path.join(os.getcwd()+'/'+folder)
            print(data_path)
            if subfolder:
                for directory in os.listdir(data_path):
                    xml_path = os.path.join(data_path, directory)
                    xml_path = r'{}'.format(xml_path)
                    xml_path = xml_path.replace("\\","/")
                    for xml_file in glob.glob(xml_path+'/*.xml'):
                        tree=ET.parse(open(xml_file))
                        root = tree.getroot()
                        image_name = root.find('filename').text
                        img_path = xml_path+'/'+image_name
                        for i, obj in enumerate(root.iter('object')):
                            difficult = obj.find('difficult').text
                            cls = obj.find('name').text
                            if cls not in names:
                                names.append(cls)
                            cls_id = names.index(cls)
                            xmlbox = obj.find('bndbox')
                            OBJECT = (str(int(float(xmlbox.find('xmin').text)))+','
                                      +str(int(float(xmlbox.find('ymin').text)))+','
                                      +str(int(float(xmlbox.find('xmax').text)))+','
                                      +str(int(float(xmlbox.find('ymax').text)))+','
                                      +str(cls_id))
                            img_path += ' '+OBJECT
                        print(img_path)
                        file.write(img_path+'\n')

    print("Class names:", names)
    with open(names_path, "w") as file:
        for name in names:
            file.write(str(name)+'\n')

def read_class_names(class_file_name):
    '''
    Input: class names file
    Output: Dictionary of Class Names
    Function: Creates dictionary of Class Names from Text file
    '''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

# This section belongs to creating YOLO model
class BatchNormalization(BatchNormalization):
    '''
    Output: Batch Normalization of Conv2D layer
    Function: Creates Batch Normalization of Conv2D layer
    '''
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True):
    '''
    Input: Layer and filter shape
    Output: Convolutional Layer
    Function: Creates convolutional layer with Batch Normalization 
    '''
    if downsample:
        input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides,
                  padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                  bias_initializer=tf.constant_initializer(0.))(input_layer)
    if bn:
        conv = BatchNormalization()(conv)
    if activate == True:
        conv = LeakyReLU(alpha=0.1)(conv)
    return conv

def residual_block(input_layer, input_channel, filter_num1, filter_num2):
    '''
    Input: Layer, input and filter shape
    Output: Residual block
    Function: Creates Residual block with Batch Normalization 
    '''
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2))

    residual_output = short_cut + conv
    return residual_output

def upsample(input_layer):
    '''
    Input: Layer
    Output: Upsampled Layer
    Function: Creates Upsampled layer
    '''
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

def darknet53(input_data):
    '''
    Input: layer specifications
    Output: DarkNet53 framework with branches
    Function: Creates DarkNet53 framework with branches 
    '''
    input_data = convolutional(input_data, (3, 3,  3,  32))
    input_data = convolutional(input_data, (3, 3, 32,  64), downsample=True)

    for i in range(1):
        input_data = residual_block(input_data,  64,  32, 64)

    input_data = convolutional(input_data, (3, 3,  64, 128), downsample=True)

    for i in range(2):
        input_data = residual_block(input_data, 128,  64, 128)

    input_data = convolutional(input_data, (3, 3, 128, 256), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 256, 128, 256)

    route_1 = input_data
    input_data = convolutional(input_data, (3, 3, 256, 512), downsample=True)

    for i in range(8):
        input_data = residual_block(input_data, 512, 256, 512)

    route_2 = input_data
    input_data = convolutional(input_data, (3, 3, 512, 1024), downsample=True)

    for i in range(4):
        input_data = residual_block(input_data, 1024, 512, 1024)

    return route_1, route_2, input_data

def YOLOv3(input_layer, num_class):
    '''
    Input: layer and number of classes
    Output: YOLOv3 framework
    Function: Creates YOLOv3 framework
    Reference: https://pylessons.com/YOLOv3-TF2-introduction/
    '''
    # We get 3 branches from darknet53
    route_1, route_2, conv = darknet53(input_layer)
    # Creating 5 sub-convolutional operators for each branch
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv = convolutional(conv, (3, 3,  512, 1024))
    conv = convolutional(conv, (1, 1, 1024,  512))
    conv_lobj_branch = convolutional(conv, (3, 3, 512, 1024))
    
    conv_lbbox = convolutional(conv_lobj_branch, (1, 1, 1024, 3*(num_class + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1,  512,  256)) 
    # Upsample uses nearest neighbour interpolation method
    conv = upsample(conv)

    conv = tf.concat([conv, route_2], axis=-1)
    conv = convolutional(conv, (1, 1, 768, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv = convolutional(conv, (3, 3, 256, 512))
    conv = convolutional(conv, (1, 1, 512, 256))
    conv_mobj_branch = convolutional(conv, (3, 3, 256, 512))

    conv_mbbox = convolutional(conv_mobj_branch, (1, 1, 512, 3*(num_class + 5)), activate=False, bn=False)

    conv = convolutional(conv, (1, 1, 256, 128))
    conv = upsample(conv)

    conv = tf.concat([conv, route_1], axis=-1)
    conv = convolutional(conv, (1, 1, 384, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv = convolutional(conv, (3, 3, 128, 256))
    conv = convolutional(conv, (1, 1, 256, 128))
    conv_sobj_branch = convolutional(conv, (3, 3, 128, 256))

    conv_sbbox = convolutional(conv_sobj_branch, (1, 1, 256, 3*(num_class +5)), activate=False, bn=False)
    return [conv_sbbox, conv_mbbox, conv_lbbox]

def Create_Yolo(input_size=416, channels=3, training=False, CLASSES=coco_classes):
    '''
    Input: Input size, channels required and classes
    Output: Keras YOLO model
    Function: Combines all layers into a keras model 
    '''
    num_class = len(read_class_names(CLASSES))
    input_layer  = Input([input_size, input_size, channels])
    conv_tensors = YOLOv3(input_layer, num_class)

    output_tensors = []
    for i, conv_tensor in enumerate(conv_tensors):
        pred_tensor = decode(conv_tensor, num_class, i)
        if training: output_tensors.append(conv_tensor)
        output_tensors.append(pred_tensor)

    Yolo = tf.keras.Model(input_layer, output_tensors)
    return Yolo


def decode(conv_output, num_class, i=0): 
    '''
    Input: Output of convolutional layers and number of classes
    Output: Tensors correspoinding to framework
    Function: Creates tensors for prediction boxes
    '''
    conv_shape       = tf.shape(conv_output)
    batch_size       = conv_shape[0]
    output_size      = conv_shape[1]

    conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + num_class))
    conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, num_class), axis=-1)

    xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
    xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2) 
    xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
    xy_grid = tf.cast(xy_grid, tf.float32)

    pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stri[i]
    pred_wh = (tf.exp(conv_raw_dwdh) * anchor[i]) * stri[i]

    pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
    pred_conf = tf.sigmoid(conv_raw_conf)
    pred_prob = tf.sigmoid(conv_raw_prob) 
    
    # Calculating prediction probability 
    return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)


def bbox_iou(boxes1, boxes2):
    '''
    Input: boxes
    Output: iou value
    Function: Calculates iou value of two boxes
    '''
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area

    return 1.0 * inter_area / union_area

def bbox_giou(boxes1, boxes2):
    '''
    Input: boxes
    Output: GIoU value
    Function: Calculates GIoU value for two boxes 
    '''
    boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                        tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
    boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                        tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    # Calculating iou value
    iou = inter_area / union_area

    enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

    enclose_area = enclose[..., 0] * enclose[..., 1]
    # Formula for GIoU value
    giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
    return giou

def compute_loss(pred, conv, label, bboxes, i=0, CLASSES=coco_classes):
    '''
    Input: Prediction with labels and layers 
    Output: giou loss, confidence loss, probability loss values
    Function: Calculates giou loss, confidence loss, probability loss values
    '''
    num_class = len(read_class_names(CLASSES))
    conv_shape  = tf.shape(conv)
    batch_size  = conv_shape[0]
    output_size = conv_shape[1]
    input_size  = stri[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + num_class))

    conv_raw_conf = conv[:, :, :, :, 4:5]
    conv_raw_prob = conv[:, :, :, :, 5:]

    pred_xywh     = pred[:, :, :, :, 0:4]
    pred_conf     = pred[:, :, :, :, 4:5]

    label_xywh    = label[:, :, :, :, 0:4]
    respond_bbox  = label[:, :, :, :, 4:5]
    label_prob    = label[:, :, :, :, 5:]

    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)
    input_size = tf.cast(input_size, tf.float32)

    bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
    giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

    iou = bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

    respond_bgd = (1.0 - respond_bbox) * tf.cast( max_iou < 0.5, tf.float32 )

    conf_focal = tf.pow(respond_bbox - pred_conf, 2)

    conf_loss = conf_focal * ( respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf)
            + respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox, logits=conv_raw_conf))

    prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1,2,3,4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1,2,3,4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1,2,3,4]))

    return giou_loss, conf_loss, prob_loss

# This section belongs to create, load and train EfficientNet Model 
def resize_image(image, size=input_size):
    '''
    Input: Image and required size
    Output: Image in resized format 
    Function: Scales the image
    '''
    w, h = image.size
    longest_side = max(w, h)
    # Scaling the image
    image = image.resize((int(w*size/longest_side), int(h*size/longest_side)), Image.BICUBIC) 
    scaled_w, scaled_h = image.size

    new_image = Image.new('RGB', (size, size))
    new_image.paste(image, ((size - scaled_w) // 2, (size - scaled_h) // 2))
    return new_image


def load_classification_data(data):
    '''
    Input: dataset type
    Output: Images in dataset along with targets
    Function: Reads the text file and loads the images with corresponding targets
    '''
    annot_path = train_annots_path if data == 'train' else test_annots_path
    images, tar = [], []
    with open(annot_path, 'r') as f:
        txt = f.readlines()
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    np.random.shuffle(annotations)
    for annotation in annotations:
        # fully parse annotations
        line = annotation.split()
        image_path, index = "", 1
        for i, one_line in enumerate(line):
            if not one_line.replace(",","").isnumeric():
                if image_path != "": image_path += " "
                image_path += one_line
            else:
                index = i
                break
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " %image_path)
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = resize_image(img)
        images.append(np.array(img))
        tar.append(line[index:][0].split(",")[4])

    return np.array(images),to_categorical(tar)

def test_video(path, model):
    '''
    Input: Video Path and EfficientNet Model
    Output: True if the fish species are detected else false
    Function: Tests the input video with EfficientNet model
    '''
    # Read video and break into frames
    cap = cv2.VideoCapture(path)
    success,image = cap.read()
    print('Total Frames :'+str(len(image)))
    if not success:
        print('Could not open file or read frame')
    else:
        images = []
        while success:
          img = Image.fromarray(image)
          img = resize_image(img)
          images.append(np.array(img))
          success,image = cap.read()
        
        images = np.array(images)
        # Pre-process Input
        images = preprocess_input(images)
        # Predicting the classes
        predictions = model.predict(images)
        argmaxes = np.argmax(predictions, axis=-1)
        #print(len(argmaxes))
        if len(argmaxes) > 1:
          return True
        else:
          return False

def models_effi(X_train, y_train,X_test, y_test, num_classes = len(read_class_names(train_classes))):
    '''
    Input: Train and validation data
    Output: Trained EfficientNet model
    Function: Trains and validates the EfficientNet model
    '''
    # Creating the model
    base_model = EfficientNetB3(weights='imagenet', include_top=False)
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    # Compiling with Adam optimiser with metric Accuracy
    model.compile(optimizer=Adam(learning_rate=3e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    # Callbacks are early stopping and checkpoints
    es = EarlyStopping(patience = 10,restore_best_weights=True)
    ckpt = ModelCheckpoint('model_effi.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
    # Training the model
    history = model.fit(X_train, y_train,callbacks=[es,ckpt], batch_size=16, epochs=10, validation_data=(X_test, y_test))
    return history, model

# This section belongs to pre-processing of input data for YOLO
def load_yolo_weights(model, weights_file):
    '''
    Input: YOLO model and weights file
    Output: YOLO weights
    Function: loads YOLO model with original weights
    '''
    # To run the process on clear session and reset layer names
    tf.keras.backend.clear_session() 
    range1 = 75 
    range2 = [58, 66, 74] 
    # Load DarkNet weigths to YOLO model
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        j = 0
        for i in range(range1):
            if i > 0:
                conv_layer_name = 'conv2d_%d' %i
            else:
                conv_layer_name = 'conv2d'
                
            if j > 0:
                bn_layer_name = 'batch_normalization_%d' %j
            else:
                bn_layer_name = 'batch_normalization'
            
            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if i not in range2:
                bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                j += 1
            else:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if i not in range2:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        assert len(wf.read()) == 0, 'failed to read all data'

def Load_Yolo_model():
    '''
    Input: YOLO model and weights file
    Output: loads YOLO model with original weights
    Function: loads YOLO model with original weights
    '''
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(f'GPUs {gpus}')
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
        
        Darknet_weights = yolo_weights
            
        if yolo_custom_weights == False:
            # Using DarkNet weights
            print("Loading Darknet_weights from:", Darknet_weights)
            yolo = Create_Yolo(input_size=input_size, CLASSES=coco_classes)
            load_yolo_weights(yolo, Darknet_weights)
        else:
            # Using Custom weights
            checkpoint = f"./checkpoints/{model_name}"
            print("Loading custom weights from:", checkpoint)
            yolo = Create_Yolo(input_size=input_size, CLASSES=train_classes)
            yolo.load_weights(checkpoint)

    return yolo
    
def image_preprocess(image, target_size, gt_boxes=None):
    '''
    Input: image and target size with Ground Truth boxes
    Output: Pre-processed image
    Function: Scaling and padding of image suppied and tailoring GT boxes
    '''
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes


def draw_bbox(image, bboxes, CLASSES=coco_classes, show_label=True, Text_colors=(255,0,0), rectangle_colors=(102,255,255), tracking=False): 
    '''
    Input: Image, bounding boxes
    Output: Bounding box around the detected objects with text
    Function: Draws Bounding box around the detected objects with prediction score
    '''
    num_class = read_clas_names(CLASSES)
    num_classes = len(num_class)
    image_h, image_w, _ = image.shape

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1: bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])
        # Draw object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            score_str = " {:.2f}".format(score)
            if tracking: score_str = " "+str(score)
            label = "{}".format(num_class[class_ind]) + score_str
            # Get Text size
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
            # Draw filled rectangle
            cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 + text_height + baseline), bbox_color, thickness=cv2.FILLED)
            # Write text on rectangle
            cv2.putText(image, label, (x1, y1 + baseline + 6), cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
    return image


def bboxes_iou(boxes1, boxes2):
    '''
    Input: boxes
    Output: iou value
    Function: Calculates iou value of bounding boxes
    '''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3):
    '''
    Input: boxes and iou threshold value
    Output: best bounding box
    Function: Selects best bounding box after Non-Max Supression
    Reference: https://github.com/bharatsingh430/soft-nms
    '''
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            iou_mask = iou > iou_threshold
            weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    '''
    Input: prediction boxes and image
    Output: Post-Processed Image
    Function: Post-Processes the images wrt threshold 
    '''
    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    org_h, org_w = original_image.shape[:2]
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    # discaeding invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    # deleting boxes with low score
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def detect_image(Yolo, image_path, input_size=416, CLASSES=coco_classes, score_threshold=0.3, iou_threshold=0.45):
    '''
    Input: YOLO model with the image path
    Output: Image with Bounding boxes 
    Function: Applies YOLO model on the input Image
    '''
    # Reading the Image
    original_image      = cv2.imread(image_path)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    # Pre-processing Image
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    # Predicting bounding Boxes
    pred_bbox = Yolo.predict(image_data)
        
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    # Post-processing Boxes
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    bboxes = nms(bboxes, iou_threshold)
    # Drawing the BB on theimage
    image = draw_bbox(original_image, bboxes, CLASSES=CLASSES)
        
    return image

def detect_video(Yolo, video_path, output_path, input_size=416, CLASSES=coco_classes, score_threshold=0.3, iou_threshold=0.45):
    '''
    Input: YOLO model with the video path
    Output: Video with Bounding boxes 
    Function: Applies YOLO model on the input Video
    '''
    times = []
    # Reading the video file and preparing the video writer for output video
    vid = cv2.VideoCapture(video_path)

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    while True:
        #Read frames in video
        _, img = vid.read()

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break
        # Pre-process the frames
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        # Predict the BB on each frame
        t1 = time.time()
        pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        # Post-processing the BB 
        bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold)
        # Drawing the BB on frames
        image = draw_bbox(original_image, bboxes, CLASSES=CLASSES)

        times.append(t2-t1)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms
        # Wring the fps data on top-left corner of frame
        image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (102, 255, 255), 2)
        
        print("FPS: {:.1f}".format(fps))
        # Writing frame to video 
        if output_path != '': out.write(image)
    cv2.destroyAllWindows()

class Dataset(object):
    '''
    Input: Dataset type
    Output: Dataset images with the annotations 
    Function: Dataset pre-processing for the YOLO model
    '''
    def __init__(self, dataset_type, test_input_size=input_size):
        # Required arguments
        self.annot_path  = train_annots_path if dataset_type == 'train' else test_annots_path
        self.input_sizes = input_size
        self.batch_size  = batch_size
        self.data_aug    = train_data_aug if dataset_type == 'train' else test_data_aug

        self.train_input_sizes = input_size
        self.strides = stri
        self.classes = read_class_names(train_classes)
        self.num_classes = len(self.classes)
        self.anchors = (anchor.T/self.strides).T
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 100
        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0


    def load_annotations(self, dataset_type):
        # loads annotations from the annotations path
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        
        for annotation in annotations:
            line = annotation.split()
            image_path, index = "", 1
            for i, one_line in enumerate(line):
                if not one_line.replace(",","").isnumeric():
                    if image_path != "": image_path += " "
                    image_path += one_line
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError("%s does not exist ... " %image_path)
            # Reads the images and stores in list
            if load_from_RAM:
                image = cv2.imread(image_path)
            else:
                image = ''
            final_annotations.append([image_path, line[index:], image])
        return final_annotations

    def __iter__(self):
        return self
    
    def Delete_bad_annotation(self, bad_annotation):
        # Deletes any bad annotations and deletes from the annotations file
        print(f'Deleting {bad_annotation} annotation line')
        bad_image_path = bad_annotation[0]
        bad_image_name = bad_annotation[0].split('/')[-1] 
        bad_xml_path = bad_annotation[0][:-3]+'xml' 

        with open(self.annot_path, "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if bad_image_name not in i:
                    f.write(i)
            f.truncate()

    def __next__(self):
        # dividing the images into batches
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice([self.train_input_sizes])
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            exceptions = False
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    try:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    except IndexError:
                        exceptions = True
                        self.Delete_bad_annotation(annotation)
                        print("IndexError, something wrong with", annotation[0], "removed this line from annotation file")

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                if exceptions: 
                    print('\n')
                    raise Exception("There were problems with dataset, I fixed them, now restart the training process.")
                self.batch_count += 1
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        # rotating the image
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        # Cropping the unwanted part of image
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        # Transforming the image using Affine transformation
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation, mAP = 'False'):
        # Parsing the annotations by using above augmentation techniques
        if load_from_RAM:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)
            
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        if mAP == True: 
            return image, bboxes
        
        image, bboxes = image_preprocess(np.copy(image), [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        # Preprocesses the truth boxes
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs

def read_clas_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
          if name.strip('\n') == 'whale':
            names[ID] = 'fish'
          else:
            names[ID] = name.strip('\n')
    return names

def voc_ap(rec, prec):
    '''
    Input: recall and precisions
    Output: Average Precision, Mean Precision and Mean Recall
    Function: Calculates Average Precision, Mean Precision and Mean Recall
    '''
    rec.insert(0, 0.0) 
    rec.append(1.0) 
    mrec = rec[:]
    prec.insert(0, 1.0) 
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def get_mAP(Yolo, dataset, score_threshold=0.25, iou_threshold=0.50, test_input_size=input_size):
    '''
    Input: YOLO model with testing dataset
    Output: mean Average Precision, Precisions, Recalls, TP's and FP's
    Function: Calculates mAP for the supplied model
    '''
    recs,precs,tps,fpss,aps = [],[],[],[],[]
    num_class = read_class_names(train_classes)

    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists('mAP'): os.mkdir('mAP')
    os.mkdir(ground_truth_dir_path)

    print(f'\ncalculating mAP {int(iou_threshold*100)}:\n')

    gt_counter_per_class = {}
    for index in range(dataset.num_samples):
        ann_dataset = dataset.annotations[index]

        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
        num_bbox_gt = len(bboxes_gt)

        bounding_boxes = []
        for i in range(num_bbox_gt):
            class_name = num_class[classes_gt[i]]
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
            # counting that object
            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                gt_counter_per_class[class_name] = 1
            bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # Sorting the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    times = []
    json_pred = [[] for i in range(n_classes)]
    for index in range(dataset.num_samples):
        ann_dataset = dataset.annotations[index]

        image_name = ann_dataset[0].split('/')[-1]
        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)
        # Reading the images and preprocessing
        image = image_preprocess(np.copy(original_image), [test_input_size, test_input_size])
        image_data = image[np.newaxis, ...].astype(np.float32)
        # Predicting the bounding boxes
        t1 = time.time()
        pred_bbox = Yolo.predict(image_data)
        t2 = time.time()
        
        times.append(t2-t1)
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        # post processing the bounding boxes 
        bboxes = postprocess_boxes(pred_bbox, original_image, test_input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold)

        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = num_class[class_ind]
            score = '%.4f' % score
            xmin, ymin, xmax, ymax = list(map(str, coor))
            bbox = xmin + " " + ymin + " " + xmax + " " +ymax
            json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})

    ms = sum(times)/len(times)*1000
    fps = 1000 / ms
    # Writing the GT boxes
    for class_name in gt_classes:
        json_pred[gt_classes.index(class_name)].sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_classes.index(class_name)], outfile)
    # Calculating Average precision of each class
    sum_AP = 0.0
    ap_dictionary = {}
    # Writing results to file
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            # Loading predictions of the classes
            predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
            predictions_data = json.load(open(predictions_file))
            # Prediction boxes to GT objects
            nd = len(predictions_data)
            tp = [0] * nd 
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # Load Prediction boxes
                bb = [ float(x) for x in prediction["bbox"].split() ]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [ float(x) for x in obj["bbox"].split() ]
                        bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            # Computing iou scores
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                            + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj
                # Assigning the predictions as TP, FP or Nan
                if ovmax >= 0.5:# if ovmax > minimum overlap
                    if not bool(gt_match["used"]):
                        # true positive
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        # update the ".json" file
                        with open(gt_file, 'w') as f:
                            f.write(json.dumps(ground_truth_data))
                    else:
                        # False positives
                        fp[idx] = 1
                else:
                    # False positives
                    fp[idx] = 1
            # Computing Precision and loss
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            tps.append(tp[-1])
            fpss.append(fp[-1])
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            aps.append(ap)
            text = class_name + " AP = {0:.3f}%".format(ap*100) #class_name + " AP = {0:.2f}%".format(ap*100)

            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            recs.append(np.array(rounded_rec, dtype=float))
            precs.append(np.array(rounded_prec, dtype=float))
            # writing results to file
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP*100, fps)
        results_file.write(text + "\n")
        print(text)
        
        return mAP*100, np.array(recs, dtype=object), np.array(precs, dtype=object), np.array(tps),np.array(fpss),np.array(aps)

def train(trainset, testset):
    '''
    Input: trainset and testset from Dataset class
    Output: trained and validated YOLO model
    Function: trains YOLO model for specified epochs on datasets
    '''
    global train_from_checkpoint
    Darknet_weights =  yolo_weights
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass

    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    writer = tf.summary.create_file_writer(log_dir)

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = 2 * steps_per_epoch
    total_steps = epochs_total * steps_per_epoch
    # Loading the YOLO model
    if transfer:
        Darknet = Create_Yolo(input_size=input_size, CLASSES=coco_classes)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    yolo = Create_Yolo(input_size=input_size, training=True, CLASSES=train_classes)
    if train_from_checkpoint:
        try:
            yolo.load_weights(f"./checkpoints/{model_name}")
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            train_from_checkpoint = False
    # Loading weights for transfer learning
    if transfer and not train_from_checkpoint:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    # Optimiser to be used
    optimizer = tf.keras.optimizers.Adam()

    # training step of themodel
    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            # Prediction results
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                # Calculating Losses
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=train_classes)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            global_steps.assign_add(1)
            if global_steps < warmup_steps:# and not transfer:
                lr = global_steps / warmup_steps * 1e-4
            else:
                lr = 1e-6 + 0.5 * (1e-4 - 1e-6)*((1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("train/loss_total", total_loss, step=global_steps)
                tf.summary.scalar("train/loss_giou", giou_loss, step=global_steps)
                tf.summary.scalar("train/loss_conf", conf_loss, step=global_steps)
                tf.summary.scalar("train/loss_prob", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # Validation step for the model
    validate_writer = tf.summary.create_file_writer(log_dir)
    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=0

            # optimizing process
            grid = 3 
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                # Calculating Losses
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=train_classes)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss
            
        return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
        
    # create second model to measure mAP
    mAP_model = Create_Yolo(input_size=input_size, CLASSES=train_classes) 
    # should be large at start
    best_val_loss = 1000 
    for epoch in range(epochs_total):
        for image_data, target in trainset:
            results = train_step(image_data, target)
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, total loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[5]))

        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(checkpoints_path, model_name))
            continue
        
        count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            prob_val += results[2]
            total_val += results[3]
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validation/val_loss", total_val/count, step=epoch)
            tf.summary.scalar("validation/giou_loss", giou_val/count, step=epoch)
            tf.summary.scalar("validation/conf_loss", conf_val/count, step=epoch)
            tf.summary.scalar("validation/prob_loss", prob_val/count, step=epoch)
        validate_writer.flush()
            
        print("\n\ntotal validation loss:{:7.2f}\n\n".format(total_val/count))
        # Saving the updated weights
        if save_checkpoint and not save_best:
            save_directory = os.path.join(checkpoints_path, model_name+"_val_loss_{:7.2f}".format(total_val/count))
            yolo.save_weights(save_directory)
        if save_best and best_val_loss>total_val/count:
            save_directory = os.path.join(checkpoints_path, model_name)
            yolo.save_weights(save_directory)
            best_val_loss = total_val/count
        if not save_best and not save_checkpoint:
            save_directory = os.path.join(checkpoints_path, model_name)
            yolo.save_weights(save_directory)

    # measure mAP of trained custom model
    try:
        # using custom weights
        mAP_model.load_weights(save_directory) 
        get_mAP(mAP_model, testset, score_threshold=0.05, iou_threshold=0.50)
    except UnboundLocalError:
        print("You don't have saved model weights to measure mAP, check save_best and save_checkpoint")

def check_video(input_path, output_path, input_size=input_size, train_classes=train_classes, show=False):
    '''
    Input: input and output path of the video supplied
    Output: Video with detections on each frame
    Function: Applies EfficientNet and YOLO model on the supplied video
    '''
    # Load EfficientNet model
    print('Starting with EfficientNet model')
    model = load_model('model_effi.h5')
    # Test the video for fish species
    p = test_video(input_path, model)
    if p == False:
      print("There are no fish species detected in the video by EfficientNet model")
      return 0 
    else:
      print("There are some fish species detected in the video by EfficientNet model \nProceeding with YOLO model")
      # Load YOLO model with custom weights
      yolo = Create_Yolo(input_size=input_size, CLASSES=train_classes)
      yolo.load_weights(f"./checkpoints/{model_name}")
      # Apply YOLO on video file
      video = detect_video(yolo, input_path, output_path, input_size=input_size, CLASSES=train_classes)
      # Show video in real time
      # Google colab has issues with cv2.imshow(), if you want to access real-time
      # feature run this code on local machine with command show=True
      if show == True:
        cap = cv2.VideoCapture(output_path)
        if (cap.isOpened()== False): 
          print("Error opening video  file")
        while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break
        cap.release()
        cv2.destroyAllWindows()