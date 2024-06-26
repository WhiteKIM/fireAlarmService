# FireAlaramService
## Overview
This project is implemented to provide fire detection information using Yolov7.<br/>
Our service can communicate the detected information to users through APIs, and based on this, it helps them to utilize the information in various fields.

## Environmnet
1. Python3
2. Yolov7

## Step - How to Use
1. Run server.py OR server.js
2. Run main_detection.py
3. Run start_api_server.py -> CORS 오류 또는 그 외의 오류가 발생함, js로 구현된 서버사용 추천 
4. web상에서 테스트하고싶은 경우 -> npm http-server활용 /test/client.html 사용

## Enviroment
1. Python3
2. html -> bootstrap4
3. javascript

## Video
[![Project_Run](https://img.youtube.com/vi/LzrBnzF2Fzw/0.jpg)](https://youtu.be/LzrBnzF2Fzw)

## Complete
1. flask를 통한 api서버
2. api서버에 탐지한 정보를 소켓을 통하여 json정보를 전달하는 기능구현
3. socketio를 활용하여 실시간 탐지영상 전달하는 서버
4. socketio를 통해 전달받은 실시간 영상을 javascript를 통하여 웹에 나타나게 하는 기능
5. json에 여러 오브젝트의 좌표를 담아서 내보내는 기능
6. api server와 탐지 프로그램간의 소켓 통신

## Update List
1. convert to Restful API
2. Flask -> FastAPI (it's slow)
3. Folder Structure
4. Improve Run Code Method
5. Cors Error
6. Remove Node.js Server

## Model Download Link
https://drive.google.com/file/d/1BqqdWtZU3k9Y18JiB1NpQX5smZ2ua-D2/view?usp=share_link <br>
Using Model before move to weights folder


## Reference
https://github.com/deshwalmahesh/yolov7-deepsort-tracking
