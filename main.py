"""
# Run Total Code
# Call this Code add some param
# 2024.06.26 - WhiteKIM
"""
import init
from multiprocessing import Process
from web.controller.DetectionController import initApiService
import argparse
from utils.detection_helpers import Detector
from main_dectection import YOLOv7_DeepSORT

if __name__ == '__main__':
    # Argument Data
    parser = argparse.ArgumentParser()
    
    # Init Shared Data
    init.init()

    # API Server Run
    api_process = Process(target=initApiService)
    api_process.start()
    
    # Object Detector Run
    source = ""
    detector = Detector(classes = [0]) # it'll detect ONLY fire
    detector.load_model('./weights/bestofbest.pt',) # pass the path to the trained weight file
    tracker = YOLOv7_DeepSORT(video=source, skip_frames=0, verbose=1, reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)

    detect_process = Process(target=tracker.track_video1)
    detect_process.start()

    # Join Process
    api_process.join()
    detect_process.join()

