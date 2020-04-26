import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
from config import shooting_result
import sys
from sys import platform
import argparse
from utils import label_map_util
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
tf.disable_v2_behavior()


def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c

def trajectory_fit(balls, height, shot):
    x = []
    y = []
    for ball in balls:
        x.append(ball[0])
        y.append(height - ball[1])

    params = curve_fit(fit_func, x, y)
    [a, b, c] = params[0]
    x_pos = np.arange(x[0], x[-1], 1)
    y_pos = []
    for i in range(len(x_pos)):
        x_val = x_pos[i]
        y_val = (a * (x_val ** 2)) + (b * x_val) + c
        y_pos.append(y_val)

    if(shot == "MISS"):
        plt.plot(x, y, 'ro')
        plt.plot(x_pos, y_pos, linestyle='-', color='red',
                 alpha=0.4, linewidth=5)  
    else:
        plt.plot(x, y, 'go')
        plt.plot(x_pos, y_pos, linestyle='-', color='green',
                 alpha=0.4, linewidth=5)

    plt.savefig("./static/detections/trajectory.jpg")

def distance(xCoor, yCoor, prev_ball):
    return ((prev_ball[0] - xCoor) ** 2 + (prev_ball[1] - yCoor) ** 2) ** (1/2)

def detect(width, height, sess, image_tensor, boxes, scores, classes, num_detections, base, prev_rim, shooting, prev_ball, first, highest, x, y, xtemp, ytemp, prev_frame, frame, head, ballinair, trace, shot_result):
    global shooting_result
    if(shot_result[0] > 0):
        shot_result[0] -= 1
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            if(classes[0][i] == 1):  # Basketball
                if(ymin < (base[0])):  # During Shooting
                    if(not shooting[0]):
                        first[0] = xCoor
                        first[1] = yCoor
                        shooting[0] = True
                        shooting[1] += 1

                    ballinair.append([xCoor, yCoor])

                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(235, 103, 193), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(235, 103, 193), thickness=-1)
                elif(ymin >= (base[0] - 30)):  # Not shooting
                    # the moment when go below basket
                    if(shooting[0] and (distance(xCoor, yCoor, prev_ball) < 100)):
                        if(xCoor >= prev_rim[0] and xCoor <= prev_rim[2]):  # shot
                            shooting_result['shot'] += 1
                            shooting_result['made'] += 1
                            shot_result[0] = 6
                            shot_result[1] = "SHOT"
                            print("shot")
                            for ballCoor in ballinair:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(82, 168, 50), thickness=-1)
                        else:  # miss
                            shooting_result['shot'] += 1
                            shooting_result['miss'] += 1
                            shot_result[0] = 6
                            shot_result[1] = "MISS"
                            print("miss")
                            for ballCoor in ballinair:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(0, 0, 255), thickness=-1)
                        trajectory_fit(ballinair, height, shot_result[1])
                        ballinair.clear()
                        shooting[0] = False

                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)

                prev_ball[0] = xCoor
                prev_ball[1] = yCoor

            if(classes[0][i] == 2):  # Rim
                cv2.rectangle(
                    trace, (prev_rim[0], prev_rim[1]), (prev_rim[2], prev_rim[3]), (255, 255, 255), 5) #cover previous hoop with white rectangle
                cv2.rectangle(frame, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)
                cv2.rectangle(trace, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)

                #display judgement after shot
                if(shot_result[0]): 
                    if(shot_result[1] == "MISS"):
                        cv2.putText(frame, shot_result[1], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
                    else:
                        cv2.putText(frame, shot_result[1], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (82, 168, 50), 8)

                prev_rim[0] = xmin
                prev_rim[1] = ymax
                prev_rim[2] = xmax
                prev_rim[3] = ymin
                if(ymin > base[0]):
                    base[0] = ymin
    combined = np.concatenate((frame, trace), axis=1)
    return combined, trace;



def getVideoStream(video_path):
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

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    trace = np.full((int(height), int(width), 3), 255, np.uint8)

    prev_rim = np.array([0, 0, 0, 0])
    base = np.array([0, 0])  # [0]value [1]set
    # shooting or not, shots, counting basket
    shooting = np.array([False, 0, False])
    prev_ball = np.array([0, 0])
    first = np.array([0, 0])
    highest = np.array([10000, 10000])
    prev_frame = np.array([None])
    head = np.array([0, 0])  # head coordinate

    x = []
    y = []
    ballinair = []
    xtemp = []
    ytemp = []
    shot_result = [0, ""]

    count = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, img = cap.read()
                count += 1
                if(count == 2):
                    count = 0
                    continue
                if ret == False:
                    break
                detection, trace = detect(width, height, sess, image_tensor, boxes, scores, classes, num_detections, base, prev_rim, shooting,
                       prev_ball, first, highest, x, y, xtemp, ytemp, prev_frame, img, head, ballinair, trace, shot_result)

                detection = cv2.resize(detection, (0, 0), fx=0.8, fy=0.8)
                frame = cv2.imencode('.jpg', detection)[1].tobytes()
                result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                yield result
    cv2.imwrite("./static/detections/trace.jpg", trace)

def detect_image(img):
    height, width = img.shape[:2]

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

        for i, box in enumerate(boxes[0]):
            # print("detect")
            if (scores[0][i] > 0.5):
                ymin = int((box[0] * height))
                xmin = int((box[1] * width))
                ymax = int((box[2] * height))
                xmax = int((box[3] * width))
                xCoor = int(np.mean([xmin, xmax]))
                yCoor = int(np.mean([ymin, ymax]))
                if(classes[0][i] == 1): #basketball
                    cv2.circle(img=img, center=(xCoor, yCoor), radius=25,
                               color=(255, 0, 0), thickness=-1)
                    cv2.putText(img, "BALL", (xCoor - 50, yCoor - 50), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 8)
                if(classes[0][i] == 2):  # Rim
                    cv2.rectangle(img, (xmin, ymax),
                                (xmax, ymin), (48, 124, 255), 10)
                    cv2.putText(img, "HOOP", (xCoor - 65, yCoor - 65),
                                cv2.FONT_HERSHEY_COMPLEX, 3, (48, 124, 255), 8)
        return img


def get_image(image_path, img_name):
    output_path = './static/detections/'
    # reading the images & apply detection 
    image = cv2.imread(image_path)
    filename = img_name
    detection = detect_image(image)

    cv2.imwrite(output_path + '{}' .format(filename), detection)
    print('output saved to: {}'.format(output_path + '{}'.format(filename)))
