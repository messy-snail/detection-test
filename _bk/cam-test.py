# -*- coding: utf-8 -*-


import jetson.inference
import jetson.utils

import numpy as np
import cv2


if __name__=='__main__':
    # cam = rs.realsense()
    # cam.open()
    cap = cv2.VideoCapture(3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    
    while (True):
        ret, frame = cap.read()
        if ret ==False:
            continue
        
        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
