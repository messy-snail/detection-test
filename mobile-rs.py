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
    
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.7)

    while (True):
        if cam.run() ==False:
            continue
        
        image = copy.deepcopy(cam.color_image)
        depth = copy.deepcopy(cam.depth_image)
        h,w,_ = image.shape
        rgba_image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).astype(np.float32)
        # move the image to CUDA:
        cuda_image = jetson.utils.cudaFromNumpy(rgba_image)
        detections = net.Detect(cuda_image, w, h)
        count = len(detections)
        # print(f"{count}개 오브젝트를 찾았습니다")

        # print("detected {:d} objects in image".format(len(detections)))
        numpyImg = jetson.utils.cudaToNumpy(cuda_image, w, h, 4)
        # now back to unit8
        result = numpyImg.astype(np.uint8)
        for detection in detections:
            if detection.ClassID==1:
                # print(detection.ROI)
                image = cv2.rectangle(result, (int(detection.ROI[0]), int(detection.ROI[1])),
                (int(detection.ROI[2]), int(detection.ROI[3])),(0,0,255), 1, cv2.LINE_AA
                )
                center_pts = (int((detection.ROI[0]+detection.ROI[2])/2), int((detection.ROI[1]+detection.ROI[3])/2))
                image = cv2.circle(image, (center_pts[0], center_pts[1]), 3, (0,0,255), -1, cv2.LINE_AA
                )
                print(f'depth[center_pts]: {depth[center_pts]}')
                zvalue = depth[center_pts]*cam.scale_factor
                cv2.putText(image, f'Depth: {zvalue:.3f}m' , (11, 40), FONT, 0.5, (32, 32, 32), 4, LINE)
                cv2.putText(image, f"Depth: {zvalue:.3f}m" , (10, 40), FONT, 0.5, (240, 240, 240), 1, LINE)
                
        # # print out timing info
        # net.PrintProfilerTimes()
        # # Display the resulting frame
        # numpyImg = jetson.utils.cudaToNumpy(input_image, w, h, 4)
        # # now back to unit8
        # result = numpyImg.astype(np.uint8)
        # # Display fps
        # fps = 1000.0 / net.GetNetworkTime()
        
        # cv2.putText(result, "FPS: " + str(int(fps)) + ' | Detecting', (11, 20), font, 0.5, (32, 32, 32), 4, line)
        # cv2.putText(result, "FPS: " + str(int(fps)) + ' | Detecting', (10, 20), font, 0.5, (240, 240, 240), 1, line)
        cv2.putText(image, "Total: " + str(count), (11, 20), FONT, 0.5, (32, 32, 32), 4, LINE)
        cv2.putText(image, "Total: " + str(count), (10, 20), FONT, 0.5, (240, 240, 240), 1, LINE)
        # # show frames
        cv2.imshow('result', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
