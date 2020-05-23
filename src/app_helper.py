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
from .utils import detect_shot, detect_image, detect_API, tensorflow_init, openpose_init
from statistics import mean
tf.disable_v2_behavior()

def getVideoStream(video_path):
    datum, opWrapper = openpose_init()
    detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    fig = plt.figure()
    #objects to store detection status
    previous = {
    'ball': np.array([0, 0]),  # x, y
    'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
        'hoop_height': 0
    }
    during_shooting = {
        'isShooting': False,
        'balls_during_shooting': [],
        'release_angle_list': [],
        'release_point': []
    }
    shooting_pose = {
        'ball_in_hand': False,
        'elbow_angle': 370,
        'knee_angle': 370,
        'ballInHand_frames': 0,
        'elbow_angle_list': [],
        'knee_angle_list': [],
        'ballInHand_frames_list': []
    }
    shot_result = {
        'displayFrames': 0,
        'release_displayFrames': 0,
        'judgement': ""
    }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36

    skip_count = 0
    with tf.Session(graph=detection_graph, config=config) as sess:
        while True:
            ret, img = cap.read()
            if ret == False:
                break
            skip_count += 1
            if(skip_count < 4):
                continue
            skip_count = 0
            detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                        num_detections, previous, during_shooting, shot_result, fig, datum, opWrapper, shooting_pose)

            detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
            frame = cv2.imencode('.jpg', detection)[1].tobytes()
            result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            yield result


    # getting average shooting angle
    shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
    shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
    shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
    shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (4 / fps), 2)

    print("avg", shooting_result['avg_elbow_angle'])
    print("avg", shooting_result['avg_knee_angle'])
    print("avg", shooting_result['avg_release_angle'])
    print("avg", shooting_result['avg_ballInHand_time'])

    plt.title("Trajectory Fitting", figure=fig)
    plt.ylim(bottom=0, top=height)
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
