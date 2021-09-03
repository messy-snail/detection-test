# -*- coding: utf-8 -*-
import realsense as rs
import gesture2

import sys

import jetson.inference
import jetson.utils

import numpy as np
import cv2
import copy

import json
import time


FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA


if __name__=='__main__':
    cam = rs.realsense()
    cam.open()

    ge = gesture2.gesture()
    net = jetson.inference.poseNet('resnet18-body', 'posenet-rs2.py', 0.15)
    # net = jetson.inference.poseNet('resnet18-hand', 'posenet-rs2.py', 0.15)
    # net = jetson.inference.poseNet('densenet121-body', 'posenet-rs2.py', 0.15)
    
    
    while (True):
        start = time.time()
        if cam.run() ==False:
            continue
                    
        image = copy.deepcopy(cam.color_image)
        depth = copy.deepcopy(cam.depth_image)
        
        h,w,_ = image.shape
               
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
        # rgba_image[:,:,3] = depth *cam.scale_factor
        cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
        poses = net.Process(cuda_image, overlay="links,keypoints")
        
        count = len(poses)
        state = ge.run(poses)            

        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        
        result = numpyImg.astype(np.uint8)
        print(result.shape)
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
        zvalue = -1
        if ge.neck_position is not None:
            zvalue = depth[int(ge.neck_position[1]),int(ge.neck_position[0])]*cam.scale_factor
        
        end = time.time()
        print(f"Time: {end-start:02f} sec")
        cv2.putText(result, f'Depth: {zvalue:.3f}m', (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'Depth: {zvalue:.3f}m', (10, 20), FONT, 0.5, (0, 0, 240), 1, LINE)
                
        cv2.imshow('res', result)
        print(state)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
