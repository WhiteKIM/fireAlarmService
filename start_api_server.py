from distutils.log import debug
from json import JSONDecodeError
from flask import Flask, Response, render_template, jsonify
import threading
import time
import json
import socket

app = Flask(__name__)
app.debug = True    #플라스크 디버깅모드 설정
app.use_reloader=False

host = '127.0.0.1'  #소켓통신을 위해 사용할 주소 및 포트
port = 12345    #소켓통신을 위해 사용할 주소 및 포트

global body

@app.route('/')
def index():
    """Video streaming home page."""
    templateData = {
            'title':'Image Streaming'
            }
    return render_template('index.html', **templateData)

# API를 통해 JSON을 전달할 함수
@app.route('/get', methods=['GET'])
def getAPI():
    # 야매로 해결
    # json파일을 읽어서 읽은 정보를 웹상에서 보여줌
    with open('result.json', 'r') as f:
        data = json.load(f)
    body = json.dumps(data)
    print(body)
    if body!=b'':
        try:
            return jsonify(json.loads(body))
        except JSONDecodeError as err:
            print(str(err))
            print('--------------------------')
            print(body)
            print('-------------------------------')
            return jsonify({'Error':'JsonDecoder error'})
    else:
        Lock.release()
        print(body)
        return jsonify({'Error':'Body is Empty Or Error'})

def GetJsonData():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen()
    client_socket, addr = server_socket.accept()
    print('Connected :'+str(addr))
    try:
        while True:
            data = client_socket.recv(1024*16)
            #print ("Raw data: ", data)
            header = data[:3]
            global body
            body = data[3:]
    except:
        print('Error')
    finally:
        server_socket.close()
    
if __name__ == '__main__':
    thread = threading.Thread(target=GetJsonData, args=())
    thread.daemon = True
    thread.start()
    time.sleep(10)
    app.run(host='0.0.0.0', threaded = True)