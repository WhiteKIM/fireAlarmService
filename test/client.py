from flask import Flask, Response
from flask_socketio import SocketIO, emit
import socketio

socketio = socketio.Client()
app = Flask(__name__)
socketio.init_app(app)

host = '127.0.0.1:25000'

imdata =''

@socketio.event
def GetData(data):
    global imdata
    data = b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+ data+ b'\r\n'
    imdata = data

@app.route('/video_feed')
def video_feed():
    return Response(imdata,mimetype='multipart/x-mixed-replace; boundary=frame')

socketio.connect(host)

while True:
    socketio.on('streaming', GetData)
