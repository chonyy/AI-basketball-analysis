from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from PIL import Image
import os
import sys
import cv2
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
        return render_template("shooting.html")


def gen(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, img = cap.read()
        if ret == False:
            break
        img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    video_path = session.get('video_path', None)
    return Response(gen(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
