import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from .config import shooting_result
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from .utils import detect_shot, detect_image, detect_API, tensorflow_init
tf.disable_v2_behavior()

def getVideoStream(video_path):
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    #objects to store detection status
    fig = plt.figure()
    previous = {
        'ball': np.array([0, 0]), #x, y
        'hoop': np.array([0, 0, 0, 0]), #xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': np.array([False]),
        'balls_during_shooting': []
    }
    shot_result = {
        'displayFrames': 0,
        'judgement': ""
    }

    skip_count = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, img = cap.read()
                skip_count += 1
                if(skip_count == 2):
                    skip_count = 0
                    continue
                if ret == False:
                    break
                detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                          num_detections, previous, during_shooting, shot_result, fig)

                detection = cv2.resize(detection, (0, 0), fx=0.8, fy=0.8)
                frame = cv2.imencode('.jpg', detection)[1].tobytes()
                result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                yield result

    plt.title("Trajectory Fitting", figure=fig)
    plt.ylim(bottom=0)
    trajectory_path = os.path.join(
        os.getcwd(), "static/detections/trajectory_fitting.jpg")
    fig.savefig(trajectory_path)
    fig.clear()
    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)

def get_image(image_path, img_name, response):
    output_path = './static/detections/'
    # reading the images & apply detection 
    image = cv2.imread(image_path)
    filename = img_name
    detection = detect_image(image, response)

    cv2.imwrite(output_path + '{}' .format(filename), detection)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))

def detectionAPI(response, image_path):
    image = cv2.imread(image_path)
    detect_API(response, image)
