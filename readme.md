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
모델 학습 : google colab 환경 -> cuda가 있는 그래픽카드를 사용하는 것을 추천함<br/>
### 실행
구동해야할 소스 코드<br/>
1. server.py OR server.js
2. main_detection.py
3. start_api_server.py
4. web상에서 테스트하고싶은 경우 -> npm http-server활용 /test/client.html 사용

## 개발언어
1. 파이썬 -> flask
2. html -> bootstrap4
3. javascript

## 실사용영상

## 구현완료
1. flask를 통한 api서버
2. api서버에 탐지한 정보를 소켓을 통하여 json정보를 전달하는 기능구현
3. socketio를 활용하여 실시간 탐지영상 전달하는 서버
4. socketio를 통해 전달받은 실시간 영상을 javascript를 통하여 웹에 나타나게 하는 기능
5. 탐지된 모든 객체의 정보가 json에 담겨 api서버를 통해 사용할 수 있음

## 수정 예정 사항
1. 적절한 모델을 아직 학습시키지 못함 --> 현재 학습중
2. 메인 웹 디자인 기능 구현중

## 참고자료
https://github.com/deshwalmahesh/yolov7-deepsort-tracking
