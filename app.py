from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from PIL import Image
import os
import sys
import cv2
from app_helper import *

app = Flask(__name__)
UPLOAD_FOLDER = './static/uploads'
#DETECTION_FOLDER = '/home/shubham-sakha/project/Repo/custom_object/FLask/static/detections'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#app.config['DETECTION_FOLDER'] = DETECTION_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        # create a secure filename
        filename = secure_filename(f.filename)
        print(filename)
        # save file to /static/uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filepath)
        f.save(filepath)
        get_image(filepath, filename)

        return render_template("uploaded.html", display_detection=filename, fname=filename)


if __name__ == '__main__':
    port = os.environ.get("PORT", 4000)
    app.run(port=port, debug=True)
