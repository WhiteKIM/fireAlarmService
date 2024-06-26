'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
FireAlarmService Main Logic
Detecting Fire And Transfer Infomation
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
from utils.tracking_helpers import read_class_names, create_box_encoder
from utils.detection_helpers import *
from collections import OrderedDict
import json
import base64
from torch.multiprocessing import Process
from flask import jsonify
import threading

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True

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
        
    # 생성된 스레드 동작을 시작
    def run(self):
        self.thread.start()

    """
    # Yolo의 동작 구조상 하나의 프레임을 한번 탐지한다. 따라서 해당 프레임이 수행된 후에 해당 탐지
    # 정보를 웹소켓이나 다른 방식을 사용해서 전송하면 될 것으로 보인다.
    """
    def track_video1(self):
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

                if self.verbose == 2:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if self.verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections

            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = vid.get(cv2.CAP_PROP_FPS)
            w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            """
            # output 영상을 웹상으로 띄어주는 코드
            # 키보드 입력으로 q가 들어오면 종료됨
            """
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break
            
            """
            # Add Data Transfer Logic
            # Using WebSocket Transfer Video Data And Json
            """
            ret, buffer = cv2.imencode('.jpg', result, encode_param)
            b64data = base64.b64encode(buffer)
        #output_video.release()
        cv2.destroyAllWindows()

"""
# WebSocket으로 데이터를 전송
# 추후 기능 개선 예정
"""
def setJsonData():
    return
    