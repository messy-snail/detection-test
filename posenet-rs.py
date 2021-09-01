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
    cam = rs.realsense()
    cam.open()

    ge = gesture.gesture()
    # net = jetson.inference.poseNet('restnet18-body', sys.argv, 0.15)

    net = jetson.inference.poseNet('densenet121-body', sys.argv, 0.15)
    
    while (True):

        if cam.run() ==False:
            continue
                    
        image = copy.deepcopy(cam.color_image)
        depth = copy.deepcopy(cam.depth_image)
        
        h,w,_ = image.shape
        # h,w = depth.shape
        print(f'h,w: {h}, {w}')
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
        rgba_image[:,:,3] = depth *cam.scale_factor
        # move the image to CUDA:
        cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
        poses = net.Process(cuda_image, overlay="links,keypoints")
        
        count = len(poses)
        
        print(f"{count}개 오브젝트를 찾았습니다.")

        angle, state, neck = ge.run(poses)
        for pose in poses:
            print(pose)
            print(pose.Keypoints)
            print('Links', pose.Links)
            

        # print out performance info
        net.PrintProfilerTimes()

        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        # now back to unit8
        result = numpyImg.astype(np.uint8)

        zvalue = -1
        if neck is not None:
            print(f'neck: {neck}')
            zvalue = depth[neck]*cam.scale_factor
        
        cv2.putText(result, "Total: " + str(count), (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, "Total: " + str(count), (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)
        
        cv2.putText(result, angle, (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, angle, (10, 40), FONT, 0.5, (240, 240, 240), 1, LINE)

        cv2.putText(result, state, (11, 60), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, state, (10, 60), FONT, 0.5, (0, 240, 0), 1, LINE)

        cv2.putText(result, f'Depth: {zvalue:.3f}m', (11, 80), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'Depth: {zvalue:.3f}m', (10, 80), FONT, 0.5, (0, 0, 240), 1, LINE)
        # # show frames
        cv2.imshow('result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
