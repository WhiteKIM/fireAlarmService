# 소개
yolov7 + deepsort 코드를 이용한 화재감지서비스. <br/>
감지 정보를 웹에서 보여줄 수 있도록 구현할 예정

## 목차
1. 환경 및 실행방법
2. 개발언어
3. 실사용 영상
4. 구현완료
5. 수정 예정 사항
6. 참고자료

## 환경
- 모델 학습 : google colab 환경 -> cuda가 있는 그래픽카드를 사용하는 것을 추천함<br/>
- 구동 환경 : LG gram 15inch 2022 -> 되도록이면 Nvidia의 그래픽카드가 탑재된 데스크탑환경에서 구동할 것을 추천함
### 실행
구동해야할 소스 코드<br/>
1. server.py OR server.js
2. main_detection.py
3. start_api_server.py -> CORS 오류 또는 그 외의 오류가 발생함, js로 구현된 서버사용 추천 
4. web상에서 테스트하고싶은 경우 -> npm http-server활용 /test/client.html 사용

## 개발언어
1. 파이썬 -> flask
2. html -> bootstrap4
3. javascript

## 실사용영상
[![Project_Run](https://img.youtube.com/vi/LzrBnzF2Fzw/0.jpg)](https://youtu.be/LzrBnzF2Fzw)

## 구현완료
1. flask를 통한 api서버
2. api서버에 탐지한 정보를 소켓을 통하여 json정보를 전달하는 기능구현
3. socketio를 활용하여 실시간 탐지영상 전달하는 서버
4. socketio를 통해 전달받은 실시간 영상을 javascript를 통하여 웹에 나타나게 하는 기능
5. json에 여러 오브젝트의 좌표를 담아서 내보내는 기능
6. api server와 탐지 프로그램간의 소켓 통신

## 수정 예정 사항
1. 메인 웹 디자인 기능 구현중

# 사용모델 링크
https://drive.google.com/file/d/1BqqdWtZU3k9Y18JiB1NpQX5smZ2ua-D2/view?usp=share_link <br>
weights폴더 내에 넣어서 사용


## 참고자료
https://github.com/deshwalmahesh/yolov7-deepsort-tracking
