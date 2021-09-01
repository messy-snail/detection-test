# -*- coding: utf-8 -*-
import realsense as rs
import gesture

import sys

import jetson.inference
import jetson.utils

import numpy as np
import cv2
import copy

import json

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA


if __name__=='__main__':
    # cam = rs.realsense()
    # cam.open()
    
    # d435
    # cap = cv2.VideoCapture(2)
    # l515
    cap = cv2.VideoCapture(3)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ge = gesture.gesture()
    # net = jetson.inference.poseNet('restnet18-body', sys.argv, 0.15)
    # densenet121-body
    # net = jetson.inference.poseNet('restnet18-body', sys.argv, 0.15)
    net = jetson.inference.poseNet('densenet121-body', sys.argv, 0.15)
    
    while (True):
        ret, frame = cap.read()
        if ret ==False:
            continue
        # if cam.run() ==False:
        #     continue
                    
        # image = copy.deepcopy(cam.color_image)
        # depth = copy.deepcopy(cam.depth_image)
        image = copy.deepcopy(frame)
        h,w,_ = image.shape
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
        # move the image to CUDA:
        cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
        poses = net.Process(cuda_image, overlay="links,keypoints")
        
        count = len(poses)
        
        print(f"{count}개 오브젝트를 찾았습니다.")

        angle, state, _ = ge.run(poses)
        for pose in poses:
            print(pose)
            print(pose.Keypoints)
            print('Links', pose.Links)
            

        # print out performance info
        net.PrintProfilerTimes()

        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        # now back to unit8
        result = numpyImg.astype(np.uint8)
        
        cv2.putText(result, "Total: " + str(count), (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, "Total: " + str(count), (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)
        
        cv2.putText(result, angle, (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, angle, (10, 40), FONT, 0.5, (240, 240, 240), 1, LINE)

        cv2.putText(result, state, (11, 60), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, state, (10, 60), FONT, 0.5, (0, 0, 240), 1, LINE)

        # # show frames
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
