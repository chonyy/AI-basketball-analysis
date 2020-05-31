<p align=center>
    <img src="./static/img/analysis.gif" width="960" height="300">
</p>

<p align=center>
    <a target="_blank" href="https://travis-ci.com/chonyy/AI-basketball-analysis" title="Build Status"><img src="https://travis-ci.com/chonyy/AI-basketball-analysis.svg?branch=master"></a>
    <a target="_blank" href="#" title="language count"><img src="https://img.shields.io/github/languages/count/chonyy/AI-basketball-analysis"></a>
    <a target="_blank" href="#" title="top language"><img src="https://img.shields.io/github/languages/top/chonyy/AI-basketball-analysis?color=orange"></a>
    <a target="_blank" href="https://img.shields.io/github/pipenv/locked/python-version/chonyy/daily-nba" title="Python version"><img src="https://img.shields.io/github/pipenv/locked/python-version/chonyy/daily-nba?color=green"></a>
    <a target="_blank" href="https://opensource.org/licenses/MIT" title="License: MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
    <a target="_blank" href="#" title="repo size"><img src="https://img.shields.io/github/repo-size/chonyy/AI-basketball-analysis"></a>
    <a target="_blank" href="http://makeapullrequest.com" title="PRs Welcome"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
</p>

> 🏀 Analyze basketball shots and shooting pose with machine learning!

This is an artificial intelligence application built on the concept of **object detection**. Analyze basketball shots by digging into the data collected from object detection. We can get the result by simply uploading files to the web App, or submitting a **POST request** to the API. Please check the [features](#features) below. There are more features coming up! Feel free to follow.

All the data for the shooting pose analysis is calculated by implementing [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose). Please note that this is an implementation only for noncommercial research use only. Please read the [LICENSE](https://github.com/chonyy/AI-basketball-analysis/blob/master/LICENSE), which is exaclty same as the [CMU's OpenPose License](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Get a copy

Get a copy of this project by simply running the git clone command.

``` git
git clone https://github.com/chonyy/AI-basketball-analysis.git
```

### Prerequisites

Before running the project, we have to install all the dependencies from requirements.txt

``` pip
pip install -r requirements.txt
```

Please note that you need a GPU with proper CUDA setup to run the video analysis, since a CUDA device is required to run OpenPose.

### Hosting

Last, get the project hosted on your local machine with a single command.

``` python
python app.py
```

#### Alternatives

This project is also hosted on [Heroku](https://ai-basketball-analysis.herokuapp.com/). However, the heavy computation of TensorFlow may cause Timeout error and crash the app (especially for video analysis). Therefore, hosting the project on your local machine is more preferable.

Please note that the shooting pose analysis won't be running on the Heroku hosted website, since a CUDA device is required to run OpenPose.

## Features

This project has three main features, [shot analysis](#shot-analysis), [shot detection](#shot-detection), [detection API](#detection-api).

### Shot analysis

#### Shot counting

<p align=center>
    <img src="./static/img/analysis_result.PNG">
</p>

Counting shooting attempts and missing, scoring shots from the input video.
Detection keypoints in different colors have different meanings listed below:
* **Blue:** Detected basketball in normal status
* **Purple**: Undetermined shot
* **Green:** Shot went in
* **Red:** Miss

#### Pose analysis

<p align=center>
    <img src="./static/img/curry_pose.gif" width="800" height="450">
</p>

Implementing [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to calculate the angle of elbow and knee during shooting.

<p align=center>
    <img src="https://2.bp.blogspot.com/_xaY-4TqFF6s/SgDgFQeGI6I/AAAAAAAAANk/I83JeZ89MOU/w1200-h630-p-k-no-nu/Basketball+Free+Throw+Side+On1-031509.jpg">
</p>

Release angle and release time are calculated by all the data collected from shot analysis and pose analysis. Please note that there will be a relatively big **error** for the release time since it was calculated as the total time when the ball is in hand.

### Shot detection

<p align=center>
    <img src="./static/img/detection.PNG">
</p>

Detection will be shown on the image. The confidence and the coordinate of the detection will be listed below.

### Detection API

<p align=center>
    <img src="./static/img/API.PNG" width="861.6" height="649.6">
</p>

Get the JSON response by submitting a **POST** request to (./detection_json) with "image" as KEY and input image as VALUE.

## Detection model

<p align=center>
    <img src="https://jkjung-avt.github.io/assets/2018-03-30-making-frcn-faster/FRCN_architecture.png" width="558" height="560.5">
</p>

The object detection model is trained with the [Faster R-CNN model architecture](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models), which includes pretrained weight on COCO dataset. Taking the configuration from the model architecture and train it on my own dataset.


## Future plans
1. Host it on azure web app service.
2. Improve the efficiency, making it executable on web app services.
