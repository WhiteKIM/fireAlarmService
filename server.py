# Node.js로 구현된 서버를 파이썬 socketio로 변환하여 구현

from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS


socketio = SocketIO(logger= True, engineio_logger = True)

def create_app():
    app = Flask(__name__)
    socketio.init_app(app, cors_allowed_origins="*")
    socketio_init(socketio)
    return app

def socketio_init(socketio):
    @socketio.on('streaming')
    def sendVideo(data):
        socketio.emit('streaming', data)

app = create_app()

if __name__ == '__main__':
    socketio.run(app ,host='127.0.0.1',port=25000, debug = True)