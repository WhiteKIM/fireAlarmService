'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os
import argparse
from queue import Empty
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *
from collections import OrderedDict
import json
import socketio
import base64
from torch.multiprocessing import Process
from flask import jsonify
import threading
import socket

host = 'http://192.168.1.37:5000'  #실제 사용할 호스트 주소
#host = 'http://127.0.0.1:25000' # 테스트용도 호스트 주소

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

sio = socketio.Client()

 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
modelList = []
jsonData = None

class Model:
    def __init__(self, name, cam, x, y, z, w):
        self.name = name
        self.cam = cam
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def getLocation(self):
        return [self.x, self.y, self.z, self.w]

    def getName(self):
        return self.name
    
    def updateLocation(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def printLocation(self):
        print('name '+str(self.name)+' '+'Location XYZW : '+str(self.x)+' '+str(self.y)+' '+str(self.z)+' '+str(self.w))

    def find(self, cam):
        if(self.cam == cam):
            return Model(self.cam, self.x, self.y, self.z, self.w)

    def getJsonInfo(self):
        jsonify = {
            'name' : str(self.name),
            'camInfo' : str(self.cam),
            'Points' : {
                'X':int(self.x),
                'Y':int(self.y),
                'Z':int(self.z),
                'W':int(self.w)
            }
        }
        return json.dumps(jsonify)

class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./IO_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()
        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker

    def __init__(self, video, skip_frames, verbose, reID_model_path:str, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="./IO_data/input/classes/coco.names",  ):
        print('생성자 호출')
            
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()
        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker
        self.skip_frames = skip_frames
        self.verbose = verbose
        self.video = video
        self.thread = threading.Thread(target=self.track_video1, args=())
        print('스레드 생성')
        
    # 생성된 스레드 동작을 시작
    def run(self):
        self.thread.start()

    def track_video1(self):
        sio.connect(host)
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(self.video))
        except:
            vid = cv2.VideoCapture(self.video)

        frame_num = 0
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            if self.skip_frames and not frame_num % self.skip_frames: continue # skip every nth frame. When every frame is not important, you can use this to fasten the process
            if self.verbose >= 1:start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            # 객체의 수를 세어주는 기능
            #count = len(names)

            #if count_objects:
                #cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain, 대략 1분에 한번정도 실행됨
            global jsonData
            jsonData = {
                'Data': []
            }
            jsonData = json.dumps(jsonData)
            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()
        
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                # class_name는 라벨명, track_id는 몇번째 객체인지에 대한 번호
                cv2.putText(frame, class_name,(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)
                #modelList.append(Model(class_name ,str ,int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                jsonify = {
                    'name' : class_name,
                    'camInfo' : 'video',
                    'Points' : {
                    'X':int(bbox[0]),
                    'Y':int(bbox[1]),
                    'Z':int(bbox[2]),
                    'W':int(bbox[3])
                    }
                }
                jsonString = json.dumps(jsonify)
                jsonString = json.loads(jsonString)
                jsonData = json.loads(jsonData)
                jsonData['Data'].append(jsonString)
                jsonData = json.dumps(jsonData)
                print(jsonData)
                '''
                    model_index = 0
                    for model in modelList:
                        if(model.getIndex()== str(track.track_id)):
                            check = False
                            break
                        model_index+=1 
                    
                if(check == False):
                    modelList[model_index].updateLocation(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                else:
                    modelList.append(Model(class_name, self.video ,int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                '''

                # 현재 존재하지 않는 모델이 있다면 제거
                # 기점은 track이 업데이트될 경우에 제거됨
                # 실시간으로 업데이트되지 않던 문제를 야매로 해결함
                '''
                self.tracker.predict()  # Call the tracker
                self.tracker.update(detections) #  updtate using Kalman Gain, 대략 1분에 한번정도 실행됨
                removeList = []
                for modelidx in range(len(modelList)):
                    count= 0
                    for track in self.tracker.tracks:
                        if str(track.track_id)== modelList[modelidx].getIndex():
                            count+=1
                    if(count==0):
                        removeList.append(modelidx)

                for rmIDX in removeList:
                    modelList.pop(rmIDX)

                '''

                
                # 디버깅을 위한 함수입니다
                # 현재 찾아낸 객체가 얼마나 존재하는지 확인하기 위한 코드입니다
                '''
                for model in modelList:
                    print(class_name, end='|')
                print('\n')
                '''

                if self.verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if self.verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                #if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                #else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            #yield frame

            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = vid.get(cv2.CAP_PROP_FPS)
            w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            #output_video = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
            #output_video.write(result)

            # output 영상을 웹상으로 띄어주는 코드
            # 키보드 입력으로 q가 들어오면 종료됨
            if True:
                #cv2.imshow('output', result)
                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            ret, buffer = cv2.imencode('.jpg', result, encode_param)
            b64data = base64.b64encode(buffer)
            ## 스트리밍을 위해 데이터를 보내는 코드
            sio.emit('streaming', b64data)
            #frame = buffer.tobytes()
            #yield frame
        #output_video.release()
        cv2.destroyAllWindows()
        sio.disconnect()

def setJsonData():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('127.0.0.1',12345))
    while True:
        '''
        jsonify = {
        "length" :len(modelList),
        "data" : []
        }
        if not modelList is Empty:
            for model in  modelList:
                jsonString = json.loads(model.getJsonInfo())
                jsonify['data'].append(jsonString)
            '''
        if not (jsonData==None):
            header = []
            header.append(0x20)
            jsonString = json.loads(jsonData)
            body = json.dumps(jsonString)
            #print('body'+str(body))
            leng = len(body)
            #print(leng)
            message= bytearray(header)
            message+= bytearray(leng.to_bytes(2, byteorder="big"))
            message+= bytes(body, 'utf-8')
            #print(message)
            client_socket.sendall(message)
        else:
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default="./IO_data/input/video/street.mp4", help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    #source = "./IO_data/input/video/street.mp4"
    source = "project.avi"
    detector = Detector(classes = [0]) # it'll detect ONLY fire
    detector.load_model('./weights/bestofbest.pt',) # pass the path to the trained weight file
    # Initialise  class that binds detector and tracker in one class
    tracker = YOLOv7_DeepSORT(video=source, skip_frames=0, verbose=1, reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
    th1 = threading.Thread(target=setJsonData, args=())
    th1.daemon = True
    th1.start()
    tracker.run()
    # output = None will not save the output video
    