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
![Project_Run](./run/detect/exp/project.avi)

## 구현완료
1. flask를 통한 api서버
2. api서버에 탐지한 정보를 소켓을 통하여 json정보를 전달하는 기능구현
3. socketio를 활용하여 실시간 탐지영상 전달하는 서버
4. socketio를 통해 전달받은 실시간 영상을 javascript를 통하여 웹에 나타나게 하는 기능

## 수정 예정 사항
1. json에 담기는 오브젝트정보가 단일 오브젝트만 저장되어 나타나는 문제
2. 적절한 모델을 아직 학습시키지 못함 --> 현재 학습중
3. 메인 웹 디자인 기능 구현중

# 사용모델 링크
https://drive.google.com/file/d/1BqqdWtZU3k9Y18JiB1NpQX5smZ2ua-D2/view?usp=share_link <br>
weights폴더 내에 넣어서 사용


## 참고자료
https://github.com/deshwalmahesh/yolov7-deepsort-tracking
