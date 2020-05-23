import os
import sys
import cv2

from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash, jsonify, abort
from werkzeug.utils import secure_filename
from PIL import Image

from src.config import shooting_result
from src.app_helper import getVideoStream, get_image, detectionAPI

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#useless key, in order to use session
app.secret_key = "super secret key" 

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/detection_json', methods=['GET', 'POST'])
def detection_json():
    if request.method == 'POST':
        response = []
        f = request.files['image']
        # create a secure filename
        filename = secure_filename(f.filename)
        print("filename", filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filepath", filepath)
        f.save(filepath)
        detectionAPI(response, filepath)
        print(response)
        try:
            return jsonify(response), 200
        except FileNotFoundError:
            abort(404)


@app.route('/sample_detection', methods=['GET', 'POST'])
def upload_sample_image():
    if request.method == 'POST':
        response = []
        filename = "sample_image.jpg"
        print("filename", filename)
        filepath = "./static/uploads/sample_image.jpg"
        print("filepath", filepath)
        get_image(filepath, filename, response)
        return render_template("shot_detection.html", display_detection=filename, fname=filename, response=response)

@app.route('/basketball_detection', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        response = []
        f = request.files['image']
        # create a secure filename
        filename = secure_filename(f.filename)
        print("filename", filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filepath", filepath)
        f.save(filepath)
        get_image(filepath, filename, response)
        return render_template("shot_detection.html", display_detection=filename, fname=filename, response=response)

@app.route('/sample_analysis', methods=['GET', 'POST'])
def upload_video():
    global shooting_result
    shooting_result['attempts'] = 0
    shooting_result['made'] = 0
    shooting_result['miss'] = 0
    if (os.path.exists("./static/detections/trajectory_fitting.jpg")):
        os.remove("./static/detections/trajectory_fitting.jpg")
    if request.method == 'POST':
        filename = "sample_video.mp4"
        print("filename", filename)
        filepath = "./static/uploads/sample_video.mp4"
        print("filepath", filepath)
        session['video_path'] = filepath
        return render_template("shooting_analysis.html")

@app.route('/shooting_analysis', methods=['GET', 'POST'])
def upload_sample_video():
    global shooting_result
    shooting_result['attempts'] = 0
    shooting_result['made'] = 0
    shooting_result['miss'] = 0
    if (os.path.exists("./static/detections/trajectory_fitting.jpg")):
        os.remove("./static/detections/trajectory_fitting.jpg")
    if request.method == 'POST':
        f = request.files['video']
        # create a secure filename
        filename = secure_filename(f.filename)
        print("filename", filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filepath", filepath)
        f.save(filepath)
        session['video_path'] = filepath
        return render_template("shooting_analysis.html")

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    stream = getVideoStream(video_path)
    return Response(stream,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/result", methods=['GET', 'POST'])
def result():
    return render_template("result.html", shooting_result=shooting_result)

#disable caching
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
