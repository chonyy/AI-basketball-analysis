from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from PIL import Image
import os
import sys
import cv2
from config import shooting_result
from app_helper import *

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/find_basketball', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['image']
        # create a secure filename
        filename = secure_filename(f.filename)
        print("filename", filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print("filepath", filepath)
        f.save(filepath)
        get_image(filepath, filename)
        return render_template("uploaded.html", display_detection=filename, fname=filename)

@app.route('/shooting_analysis', methods=['GET', 'POST'])
def upload_video():
    global shooting_result
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
        return render_template("shooting.html", result="from initial")

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    session['result'] = "fuck"
    stream = getVideoStream(video_path)
    return Response(stream,
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    return render_template("shooting.html", result="shit")

@app.context_processor
def context_processor():
    return dict(result="from processor")

@app.route("/result", methods=['GET', 'POST'])
def result():
    return render_template("result.html", shot=shooting_result['shot'], made=shooting_result['made'], miss=shooting_result['miss'])


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
