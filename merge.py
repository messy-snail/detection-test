# -*- coding: utf-8 -*-
import numpy as np
import cv2
import jetson.inference
import jetson.utils

import sys
import copy

import realsense as rs

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA
    

if __name__=='__main__':
    cam = rs.realsense()
    cam.open()
    
    detection_net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.7)
    detection_net.SetOverlayAlpha(50);

    ge = gesture.gesture()
    posenet = jetson.inference.poseNet('restnet18-body', sys.argv, 0.15)

    while (True):
        if cam.run() ==False:
            continue
        
        image = copy.deepcopy(cam.color_image)
        depth = copy.deepcopy(cam.depth_image)
        h,w,_ = image.shape
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
        # move the image to CUDA:
        cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
        detections = detection_net.Detect(cuda_image, w, h)
        count = len(detections)
        # print(f"{count}개 오브젝트를 찾았습니다")

        # print("detected {:d} objects in image".format(len(detections)))
        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        # now back to unit8
        result = numpyImg.astype(np.uint8)
        for detection in detections:
            if detection.ClassID==1:


                roi_img = copy.deepcopy(image[int(detection.ROI[1]):int(detection.ROI[3]), int(detection.ROI[0]):int(detection.ROI[2])]) 
                cv2.imshow('roi', roi_img)

  
        # # show frames
        cv2.imshow('result', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
