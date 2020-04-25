import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import sys
from sys import platform
import argparse
from utils import label_map_util
from utils import visualization_utils as vis_util
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
tf.disable_v2_behavior()


def detect(image_path):
    img = cv2.imread(image_path)

    MODEL_NAME = 'inference_graph_new2'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = 'training/labelmap.pbtxt'

    NUM_CLASSES = 2

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    with tf.Session(graph=detection_graph) as sess:
        img_expanded = np.expand_dims(img, axis=0)
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: img_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            # min_score_thresh=.95,
            use_normalized_coordinates=True,
            line_thickness=8)
        return img


def get_image(image_path, img_name):
    classes_path = './data/labels/coco.names'
    weights_path = './weights/yolov3.tf'
    tiny = False
    size = 416
    output_path = './static/detections/'
    num_classes = 80

    # load in weights and classes
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights_path).expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(classes_path).readlines()]
    print('classes loaded')
    # reading the images & apply detection with loaded weight file
    image = cv2.imread(image_path)
    filename = img_name
    img_raw = tf.image.decode_jpeg(
        open(image_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, size)

    t1 = time.time()
    boxes, scores, classes, nums = yolo(img)
    t2 = time.time()
    print('time: {}'.format(t2 - t1))

    # print('detections:')
    # print(nums[0])
    # for i in range(nums[0]):
    #     print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
    #                                 np.array(scores[0][i]),
    #                                 np.array(boxes[0][i])))
    # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(output_path + '{}' .format(filename), image)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))
