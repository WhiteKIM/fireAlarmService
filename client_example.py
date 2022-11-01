import numpy as np
import cv2
import socketio
import base64

imdata = 0
host = 'http://127.0.0.1:25000'
sio = socketio.Client()

@sio.event
def Streaming(data):
    data = cv2.imdecode(np.frombuffer(base64.b64decode(data), dtype='uint8'), cv2.IMREAD_COLOR)
    global imdata
    imdata = data

sio.connect(host)

while True:
    sio.on('streaming', Streaming)
    cv2.imshow('test', imdata)
    if(cv2.waitKey(1) > 0):
        break

cv2.destroyAllWindows()
sio.disconnect()