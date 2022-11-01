from concurrent.futures import thread
from distutils.log import debug
from json import JSONDecodeError
from shutil import which
from socket import socket
import threading
from detection_helpers import *
from tracking_helpers import *
from custom_bridge_wrapper import *
from PIL import Image
from flask import Flask, Response, render_template, jsonify, abort, redirect, url_for
#from flask_ngrok import run_with_ngrok
import datetime

app = Flask(__name__)
app.debug = True

@app.route('/')
def index():
    """Video streaming home page."""
    now = datetime.datetime.now()
    timeString = now.strftime("%Y-%m-%d %H:%M")
    templateData = {
            'title':'Image Streaming',
            'time': timeString
            }
    return render_template('index.html', **templateData)

# API를 통해 JSON을 전달할 함수
@app.route('/get', methods=["GET"])
def getAPI():
    jsonify = {
        "length" : len(modelList),
        "data" : []
    }

    for model in modelList:
        jsonString = json.loads(model.getJsonInfo())
        jsonify['data'].append(jsonString)
    return jsonify
    
def gen():
    while True:
        for frame in tracker.track_video(opt.source, skip_frames = 0, verbose=1):
            yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'
            yield frame
            yield b'\r\n'

# 테스트용 함수
# 웹상에서 연산을 수행한 결과값이 잘 나오는지를 확인하는 코드
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./IO_data/input/video/street.mp4", help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    detector = Detector(classes = [0]) # it'll detect ONLY fire
    detector.load_model('./weights/yolov7x.pt',) # pass the path to the trained weight file
    # Initialise  class that binds detector and tracker in one class
    tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
    # output = None will not save the output video
    app.run()
    
    