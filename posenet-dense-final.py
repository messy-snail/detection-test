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

KEEP_FRAME_NUM = 5

if __name__=='__main__':
    cam = rs.realsense()
    cam.open()

    ge = gesture2.gesture()
    # net = jetson.inference.poseNet('resnet18-body', 'posenet-rs2.py', 0.15)
    # net = jetson.inference.poseNet('resnet18-hand', 'posenet-rs2.py', 0.15)
    net = jetson.inference.poseNet('densenet121-body', 'posenet-rs2.py', 0.15)
    
    prev_cmd = ''
    frame_counter = 0
    final_cmd='N/A'
    
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
        poses = net.Process(cuda_image, overlay="links,keypoints,box")
        
        count = len(poses)
        ge.run(poses, depth, cam.scale_factor)

        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        
        result = numpyImg.astype(np.uint8)
        
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)
        zvalue = -1
        if ge.neck_position is not None:
            zvalue = depth[int(ge.neck_position[1]),int(ge.neck_position[0])]*cam.scale_factor

        end = time.time()
        # print(f"Time: {end-start:02f} sec")
        cv2.putText(result, f'Depth: {zvalue:.3f}m', (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'Depth: {zvalue:.3f}m', (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)

        for i in range(0, count):
            id_img = result[int(poses[i].ROI[1]):int(poses[i].ROI[3]), int(poses[i].ROI[0]):int(poses[i].ROI[2]), :]
            if id_img.shape[0] !=0 and id_img.shape[1] !=0:
                cv2.putText(id_img, f'{i}', (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
                cv2.putText(id_img, f'{i}', (10, 40), FONT, 0.5, (240, 240, 240), 1, LINE)   
        
        
        operator=None
        if count>0:
            index = ge.registration_id
            if index>=0:
                operator = result[int(poses[index].ROI[1]):int(poses[index].ROI[3]), int(poses[index].ROI[0]):int(poses[index].ROI[2]), :]
                cmd = ge.total_state_dict[index]
                
                print(f'operator.shape: {operator.shape}')
                if operator.shape[0] !=0 and operator.shape[1] !=0:
                    if cmd == 'GO' or cmd == 'STOP' or cmd=='FOLLOW' or cmd=='LEFT'or cmd=='RIGHT':
                        if prev_cmd==cmd:
                            frame_counter+=1
                        else:
                            frame_counter=0     
                            # final_cmd='N/A'                       
                        
                        if frame_counter>KEEP_FRAME_NUM:
                            frame_counter=0     
                            final_cmd = cmd            
                        prev_cmd = cmd  
                                       
                        cv2.putText(operator, final_cmd, (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
                        cv2.putText(operator, final_cmd, (10, 20), FONT, 0.5, (0, 240, 0), 1, LINE)
                    # else:
                    #     cv2.putText(operator, 'N/A', (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
                    #     cv2.putText(operator, 'N/A', (10, 20), FONT, 0.5, (0, 0, 240), 1, LINE)                            
                    
                    # cv2.imshow('Operator', operator)
            # else:
            #     cv2.putText(result, 'No Operator', (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
            #     cv2.putText(result, 'No Operator', (10, 40), FONT, 0.5, (0, 0, 240), 1, LINE)                
        cv2.putText(operator, final_cmd, (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(operator, final_cmd, (10, 20), FONT, 0.5, (0, 240, 0), 1, LINE)
        cv2.imshow('Pose Candidate', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
