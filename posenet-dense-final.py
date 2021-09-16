# -*- coding: utf-8 -*-
import realsense as rs
import gesture

import jetson.inference
import jetson.utils

import numpy as np
import cv2
import copy


import time


FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE = cv2.LINE_AA

KEEP_FRAME_NUM = 5

if __name__=='__main__':
    cam = rs.realsense()
    cam.open()

    ge = gesture.gesture()
    # net = jetson.inference.poseNet('resnet18-body', 'posenet-rs2.py', 0.15)
    net = jetson.inference.poseNet('densenet121-body', 'posenet-rs2.py', 0.20)

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

        numpy_img = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        
        result = numpy_img.astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGBA2BGR)

        xyz=[-1, -1, -1]
        if ge.neck_position is not None:
            xyz = cam.get_position(int(ge.neck_position[0]), int(ge.neck_position[1]))

        end = time.time()
        cv2.putText(result, f'X: {xyz[0]:.3f}m, Y: {xyz[1]:.3f}m, Z: {xyz[2]:.3f}m', (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'X: {xyz[0]:.3f}m, Y: {xyz[1]:.3f}m, Z: {xyz[2]:.3f}m', (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)

        cv2.putText(result, f'L Shoulder: {ge.left_angle[0]:.2f}, Elbow: {ge.left_angle[1]:.2f}', (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'L Shoulder: {ge.left_angle[0]:.2f}, Elbow: {ge.left_angle[1]:.2f}', (10, 40), FONT, 0.5, (180, 240, 180), 1, LINE)

        cv2.putText(result, f'R Shoulder: {ge.right_angle[0]:.2f}, Elbow: {ge.right_angle[1]:.2f}', (11, 60), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'R Shoulder: {ge.right_angle[0]:.2f}, Elbow: {ge.right_angle[1]:.2f}', (10, 60), FONT, 0.5, (240, 180, 180), 1, LINE)

        for i in range(0, count):
            id_img = result[int(poses[i].ROI[1]):int(poses[i].ROI[3]), int(poses[i].ROI[0]):int(poses[i].ROI[2]), :]
            if id_img.shape[0] !=0 and id_img.shape[1] !=0:
                cv2.putText(id_img, f'id: {i}', (id_img.shape[1]-50, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
                cv2.putText(id_img, f'id: {i}', (id_img.shape[1]-51, 20), FONT, 0.5, (180, 180, 240), 1, LINE)   

        operator=None
        red_color = (50,50,240)
        green_color =(50,240,50)

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

                        if frame_counter>KEEP_FRAME_NUM:
                            frame_counter=0
                            final_cmd = cmd
                        prev_cmd = cmd

                        cv2.putText(result, f'Last command: {final_cmd}', (11, 80), FONT, 0.5, (32, 32, 32), 4, LINE)
                        cv2.putText(result, f'Last command: {final_cmd}', (10, 80), FONT, 0.5, (0, 240, 0), 1, LINE)
                    if final_cmd =='FOLLOW':
                        result = cv2.rectangle(result, (int(poses[index].ROI[0]), int(poses[index].ROI[1])),
                        (int(poses[index].ROI[2]), int(poses[index].ROI[3])), red_color, 2, cv2.LINE_AA)

                    else:
                        result = cv2.rectangle(result, (int(poses[index].ROI[0]), int(poses[index].ROI[1])),
                        (int(poses[index].ROI[2]), int(poses[index].ROI[3])), green_color, 2, cv2.LINE_AA)

        cv2.putText(result, f'Last command: {final_cmd}', (11, 80), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(result, f'Last command: {final_cmd}', (10, 80), FONT, 0.5, (0, 240, 0), 1, LINE)

        cv2.imshow('Pose Candidate', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
